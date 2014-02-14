import sys
from qt import *
import numpy as _N
import display
from pyhrf.ndarray import *

debug = 1

image0_data = \
    "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d" \
    "\x49\x48\x44\x52\x00\x00\x00\x14\x00\x00\x00\x20" \
    "\x08\x06\x00\x00\x00\x12\x62\x58\xb6\x00\x00\x00" \
    "\xf1\x49\x44\x41\x54\x48\x89\xed\xd4\x21\x4e\x03" \
    "\x41\x14\x06\xe0\x6f\xca\x3a\x50\x10\xc2\x0d\x90" \
    "\x55\x88\x5e\x00\x8d\x24\x41\xa0\xe1\x12\x04\xc5" \
    "\x05\x10\x5c\x83\x70\x00\x3c\xb2\x58\x10\x08\x64" \
    "\x31\x88\xa6\xa4\x84\x87\xd8\x6d\xb2\x6d\x80\x59" \
    "\x60\xc5\x36\xe9\x4b\x9e\x99\xfc\xfb\xcd\xec\xcc" \
    "\x64\x44\x84\x9f\x1a\x9b\x18\xe1\x3c\x97\x8d\x08" \
    "\x39\xac\x87\x5b\x7c\xe0\x1d\x83\xff\x82\x17\x98" \
    "\x20\x2a\x74\x84\x9d\x3f\x81\x38\xc0\xb4\xc2\x66" \
    "\xfd\x86\x3b\x14\xbf\x02\xb1\x8b\xf1\x02\x36\xeb" \
    "\x09\x2e\x1b\x83\xd8\xc0\xe3\x17\xab\xab\xf7\x14" \
    "\x47\x4d\xc1\xeb\x6a\xbf\xbe\xc3\xea\xbf\xdf\x5f" \
    "\xfc\xbe\x50\xab\x94\xd2\x5e\x35\xfb\x8d\xf9\xda" \
    "\xc7\x03\x9e\x6a\x63\x81\x43\xdc\xcf\x25\x1b\xdd" \
    "\xad\x12\x3a\x69\x92\xed\x69\xb9\xba\x0f\x16\x29" \
    "\xa5\xe3\x06\xb9\x75\x0c\x52\x4a\xe3\x2c\x88\x33" \
    "\x5c\x65\x72\x6b\xca\xfb\xb9\x9d\xc9\x9d\xc2\xb0" \
    "\xad\x53\xc6\xb0\xfb\x87\xb2\x02\x3b\x08\x16\xf9" \
    "\x08\xe8\x2b\x5f\xea\x76\xc0\x88\x78\x6d\x38\xf1" \
    "\x12\xec\xe1\x0a\xec\x20\x98\xf0\x8c\x97\x96\xbc" \
    "\xad\x4f\xa5\x83\xa5\xf1\x3d\xdf\x60\x6b\x00\x00" \
    "\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82"
image1_data = \
    "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d" \
    "\x49\x48\x44\x52\x00\x00\x00\x14\x00\x00\x00\x1c" \
    "\x08\x06\x00\x00\x00\x61\xda\x9f\x60\x00\x00\x01" \
    "\x22\x49\x44\x41\x54\x48\x89\xed\x94\xbf\x4a\xc4" \
    "\x40\x10\x87\xbf\x91\x14\x36\x57\xdd\x43\x58\x28" \
    "\x08\x82\xd8\xd8\x0b\x82\xb5\xbd\x08\x62\xe3\x53" \
    "\xf8\x06\x82\x3e\x83\x8d\xa5\xdd\x59\x89\x85\x5d" \
    "\xe0\x9a\xb3\x12\x51\xd0\x56\x9b\xf3\x5f\x7e\x16" \
    "\x37\x07\x31\x5c\x76\x56\xb0\x88\xe0\x42\x20\xd9" \
    "\xfd\xe6\x9b\xd9\x64\xb2\x26\x89\x68\x98\xd9\x16" \
    "\x30\x92\x74\x13\xb1\x73\xa1\x6d\x32\x8e\x80\xcd" \
    "\x1c\x30\x57\x98\x3d\xfe\x85\x1d\x14\x1a\x70\x9d" \
    "\xc1\x2d\x03\x8f\xc0\x53\x04\x16\xc0\x3c\xb0\x1e" \
    "\x70\x43\xe0\x04\x38\x0e\xb8\xcb\x02\x90\xa4\xe7" \
    "\x14\x65\x66\x15\x30\xce\xe0\xd4\xfd\x8f\xd2\x7d" \
    "\x61\x51\x7f\x30\xb3\x15\x60\x6f\x46\xa2\x3e\xb0" \
    "\x6d\x66\x8b\xb5\x39\x01\x77\x92\x0e\x9b\xd2\x52" \
    "\x12\x7e\x2e\x1a\x30\x00\x2a\x0f\x48\x5d\x1f\xc0" \
    "\xda\x34\xd6\xe3\xcb\x6f\x42\x9f\xec\x33\x69\xe2" \
    "\xcf\x84\xec\x15\x38\xa8\xc7\xb5\x0a\x7d\x61\x15" \
    "\x78\x6f\x91\x8d\x81\xb3\x66\x4c\x52\xe8\x8b\xfb" \
    "\xc0\xdb\x8c\x6d\xde\x02\xbd\x1f\x0b\x1d\x38\xf5" \
    "\x8a\xea\x5b\x5d\x4a\xf0\x65\xd4\x36\x3b\xc0\xbd" \
    "\xdf\x57\xc0\xae\xa4\x61\x10\xd3\x5e\xa1\x67\x5d" \
    "\xf0\xf7\x79\x9e\xe2\xa6\x15\x16\xa9\x4c\x00\x92" \
    "\x46\x66\xb6\x01\x5c\x45\x2c\x34\x1a\x3b\x21\xbd" \
    "\xc8\xe1\xe0\x2f\xfc\xcb\x06\x3c\x00\x2f\xbf\xe4" \
    "\xeb\x7d\x01\x4b\x4c\x12\x73\xed\x64\xdd\x8f\x00" \
    "\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82"




class AxisSlicer(QWidget):

    def __init__(self, axisName, axisDomain, parent = None, name = None, fl = 0):
        QWidget.__init__(self,parent,name,fl)

        self.axisName = axisName
        self.axisDomain = axisDomain

        self.image0 = QPixmap()
        self.image0.loadFromData(image0_data,"PNG")
        self.image1 = QPixmap()
        self.image1.loadFromData(image1_data,"PNG")

        if not name:
            self.setName("AxisSlicer")

        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred,
                                       QSizePolicy.Preferred,0,0,
                                       self.sizePolicy().hasHeightForWidth()))

        AxisSlicerLayout = QVBoxLayout(self,0,0,"AxisSlicerLayout")

        self.frame11 = QFrame(self,"frame11")
        f11sp = self.frame11.sizePolicy()
        self.frame11.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,
                                               QSizePolicy.Fixed,0,0,
                                               f11sp.hasHeightForWidth()))
        self.frame11.setMinimumSize(QSize(95,205))
        self.frame11.setFrameShape(QFrame.StyledPanel)
        self.frame11.setFrameShadow(QFrame.Raised)

        self.axisLabel = QLabel(self.frame11,"axisLabel")
        self.axisLabel.setGeometry(QRect(10,10,80,17))
        self.axisLabel.setText(self.axisName)
        
        self.valueLabel = QLabel(self.frame11,"valueLabel")
        self.valueLabel.setGeometry(QRect(10,180,80,17))
        self.valueLabel.setText(str(self.axisDomain[0]))


        self.slider1 = QSlider(self.frame11,"slider1")
        self.slider1.setGeometry(QRect(10,30,24,140))
        self.slider1.setOrientation(QSlider.Vertical)
        self.slider1.setMaxValue(len(self.axisDomain)-1)

        QToolTip.add(self, str(self.axisDomain[0]))
        #AxisSlicerLayout.addWidget(self.frame11)

        self.talkBox = QCheckBox(self.frame11,"talkBox")
        self.talkBox.setGeometry(QRect(40,30,40,40))
        self.talkBox.setPixmap(self.image0)

        self.listenBox = QCheckBox(self.frame11,"listenBox")
        self.listenBox.setGeometry(QRect(40,135,40,50))
        self.listenBox.setPixmap(self.image1)
        self.listenBox.setChecked(not (self.name() == 'maskSlicer'))
        AxisSlicerLayout.addWidget(self.frame11)

        self.languageChange()
        
        self.resize(QSize(102,137).expandedTo(self.minimumSizeHint()))
        self.clearWState(Qt.WState_Polished)

        self.connect(self.slider1,SIGNAL("valueChanged(int)"),
                     self.slider1_valueChanged)

        self.connect(self.slider1, SIGNAL("valueChanged(int)"),
                     self.declareSliceChange)
        
    def languageChange(self):
        self.setCaption(self.__tr("browser"))
        #self.talkBox.setText(QString.null)
        #self.listenBox.setText(QString.null)

    def declareSliceChange(self, i):
        if debug: print 'declaring slice change ...', i, self
        if self.talkBox.isChecked():
            if debug: print 'talkbox is checked -> spreading the news !'
            self.emit(PYSIGNAL('declareSliceChange'), (i,))

    def recieveSliceChange(self, i):
        if debug: print 'recieving slice change ...', i, self
        if self.listenBox.isChecked():
            if debug: print 'listen box is checked -> I will listen to that !'
            self.slider1.setValue(i)
            #self.slider1_valueChanged(i)
        
    def value(self):
        return self.slider1.value()

    def getSliceFromRealValue(self, v):
        if debug:
            print 'searching index of v:', v
            print 'in :', self.axisDomain
        idx = _N.where(self.axisDomain==v)
        if debug:
            print '_N.where(self.axisDomain==v) :', idx
        if len(idx[0])>0:
            if debug: print ' -> ', idx[0][0]
            return idx[0][0]
        else:
            if debug: print '-> not found !!'
            return None

    def updateValueLabel(self, a0):
        # TODO : put number in form a.eb
        if not _N.isreal(self.axisDomain[a0]):
            self.valueLabel.setText(self.axisDomain[a0])
        else:
            self.valueLabel.setText('%1.3g' %self.axisDomain[a0])
        QToolTip.add(self, str(self.axisDomain[a0]))


    def slider1_valueChanged(self,a0):
        if debug: print "AxisSlicer.slider1_valueChanged(%d) ..." %a0
        if debug: print "-> ", self
        self.updateValueLabel(a0)
        self.emit(PYSIGNAL("sliceChanged"),(self.axisName,a0))

    def __tr(self,s,c = None):
        return qApp.translate("AxisSlicer",s,c)


def all_perms(seq):
    if len(seq) <=1:
        yield seq
    else:
        for perm in all_perms(seq[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + seq[0:1] + perm[i:]

class IndexHolder(QObject):
    def __init__(self, idx,**args):
        QObject.__init__(self)
        self.idx = idx
    def __call__(self, *args):
        if debug:print 'IndexHolder.__call__'
        if debug:print (self.idx,)+args
        self.emit(PYSIGNAL("transferedSignal"),(self.idx,)+args)


        
class ExclusiveBoxSelector(QWidget):
    def __init__(self, items, nbMaxSelectors=3, parent = None,
                 name = None,fl = 0):
        QWidget.__init__(self,parent,name,fl)
        
        if not name:
            self.setName("ExclusiveBoxSelector")

        self.itemList = items
        if nbMaxSelectors > len(items):
            self.nbSelectors = len(items)
        else:
            self.nbSelectors = nbMaxSelectors
        
        self.nbShown = nbMaxSelectors
        
        self.setSizePolicy(QSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum,0,0,
                                       self.sizePolicy().hasHeightForWidth()))

        ExclusiveBoxSelectorLayout = QHBoxLayout(self,0,0,
                                                 "ExclusiveBoxSelectorLayout")

        self.axisSelecFrame = QFrame(self,"axisSelecFrame")
        self.axisSelecFrame.setFrameShape(QFrame.NoFrame)
        self.axisSelecFrame.setFrameShadow(QFrame.Plain)
        axisSelecFrameLayout = QHBoxLayout(self.axisSelecFrame,0,0,
                                           "axisSelecFrameLayout")
        self.boxLists = []
        self.boxes = []
        self.iBoxes = []
        for i in xrange(self.nbSelectors):
            box = QComboBox(False,self.axisSelecFrame,"axis1Box")
            self.boxes.append(box)
            axisSelecFrameLayout.addWidget(box)
            iBox = IndexHolder(i)
            self.iBoxes.append(iBox)
            self.connect(box, SIGNAL("activated(const QString&)"), iBox)
            self.connect(iBox, PYSIGNAL("transferedSignal"),
                         self.comboBoxChanged)
            l = self.itemList[i:i+1] + self.itemList[self.nbSelectors:]
            self.boxLists.append(l)

        if debug: print 'boxLists at init:'
        if debug: print self.boxLists

        self.setShownNumber(self.nbShown)
        
        self.swapButton = QPushButton(self.axisSelecFrame,"swapButton")
        axisSelecFrameLayout.addWidget(self.swapButton)
        self.connect(self.swapButton, SIGNAL("clicked()"), self.swapSelections)
        ExclusiveBoxSelectorLayout.addWidget(self.axisSelecFrame)

        self.languageChange()

        self.resize(QSize(351,54).expandedTo(self.minimumSizeHint()))
        self.clearWState(Qt.WState_Polished)

    def setSelection(self, selection):
        i = 0
        if debug: print 'ExclusiveBoxSelectorLayout.setSelection to ', selection
        sItems = set(self.itemList)
        if debug:print '*** self.boxLists :', len(self.boxLists)
        for s in selection:
            if debug:print '******* i :', i
            self.boxLists[i] = [s] + list(sItems.difference(selection))
            i += 1
        for i in xrange(len(selection)+1,len(self.boxLists)):
            self.boxLists[i] = list(sItems.difference(selection))
        
        if debug: print '-> ', self.boxLists
        self.permutations = all_perms(self.getSelections())
        self.permutations.next()
        self.fillBoxes()
            

            
    def fillBoxes(self):
        if debug: print 'ExclusiveBoxSelector.fillBoxes ...'
        for i in xrange(self.nbShown):
            if debug: print 'filling Box %d ...' %i
            box = self.boxes[i]
            box.clear()
            for s in self.boxLists[i]:
                if debug: print ' +', s
                box.insertItem(s)

                
    def comboBoxChanged(self, idx, new):
        if debug: print 'ExclusiveBoxSelector.comboBoxChanged ...'
        new = str(new)
        if debug: print 'idx = %d, new = %s' %(idx,new)
        if debug: print ' updating box of index:', idx
        if debug: print '  previous list :', self.boxLists[idx]
        bl = self.boxLists[idx]
        if bl[0] == new:
            return
        prev = bl.pop(0)
        bl.append(prev)
        bl.remove(new)
        bl.insert(0, new)
        if debug: print '  new list :', self.boxLists[idx]
        if debug: print ' updating boxes of index:', range(idx)+\
           range(idx+1,self.nbSelectors)
        for i in range(idx)+range(idx+1,self.nbSelectors):
            if debug: print ' updating box of index:', i
            if debug: print '  previous list :', self.boxLists[i]
            if new in self.boxLists[i]:
                self.boxLists[i].remove(new)
            if prev not in self.boxLists[i]:
                self.boxLists[i].append(prev)
            if debug: print '  new list :', self.boxLists[i]
        self.fillBoxes()
        self.permutations = all_perms(self.getSelections())
        self.permutations.next()
        self.emit(PYSIGNAL("selectionChanged"),(self.getSelections(),))
        
    def setShownNumber(self, newns):
        if debug : print 'ExclusiveBoxSelector.setShownNumber(%s)' %str(newns)
        if newns == self.nbShown:
            return

        prevSelec = self.getSelections()
        if debug : print 'prevSelec'
        if debug : print prevSelec
        sItems = set(self.itemList)
        
        newSelec = [ l[0] for l in self.boxLists[:newns] ]
        if debug : print 'newSelec'
        if debug : print newSelec
        #if debug : print '[:newns-self.nbShown]=[:%d-%d]=[:%d]' \
        #   %(newns, self.nbShown, newns-self.nbShown)
        #if newns > self.nbShown:
        #    newSelec += list(sItems.difference(newSelec))[:newns-self.nbShown]
            
        #if debug : print 'newSelec2'
        #if debug : print newSelec
        self.nbShown = newns
        i = 0
        for box in self.boxes:
            if i < newns: box.show()
            else: box.hide()
            i += 1
        self.setSelection(newSelec)
        self.emit(PYSIGNAL("selectionChanged"),(self.getSelections(),))
        
    def getSelections(self):
        return [bl[0] for bl in self.boxLists[:self.nbShown]]
        
    def swapSelections(self):
        if debug: print 'ExclusiveBoxSelector.swapSelections ...'
        if self.nbShown == 1:
            return
        previousOrder = self.getSelections()
        try:
            newSelections = self.permutations.next()
        except StopIteration, e:
            self.permutations = all_perms(self.getSelections())
            self.permutations.next()
            newSelections = self.permutations.next()
        i = 0
        if debug: print 'newSelections :', newSelections
        for s in newSelections:
            if debug: print ' previous list of %d :' %i, self.boxLists[i]
            self.boxLists[i][0] = s
            if debug: print ' new list of %d :' %i, self.boxLists[i]
            i += 1
        self.fillBoxes()
        if debug: print 'ExclusiveBoxSelector.swap(%s, %s) ...' \
           %(str(previousOrder), str(newSelections))
        self.emit(PYSIGNAL("selectionSwapped"),(previousOrder,newSelections))
        
    def languageChange(self):
        self.setCaption(self.__tr("Form1"))
        self.swapButton.setText(self.__tr("swap"))


    def __tr(self,s,c = None):
        return qApp.translate("ExclusiveBoxSelector",s,c)


class AxisBrowser(QWidget):

    MODE_1D = 0
    MODE_2D = 1
    MODE_3D = 2
    MODE_4D = 3
    
    def __init__(self, cuboidView, parent = None,name = None,modal = 0,fl = 0):
        QWidget.__init__(self,parent,name,fl)

        self.cuboidView = cuboidView
        self.mode = len(self.cuboidView.current_axes) - 1
        if debug: print 'Init of AxisBrowser.mode = ', self.mode
        if not name:
            self.setName("AxisBrowser")
            
        self.setBaseSize(QSize(200,350))

        AxisBrowserLayout = QVBoxLayout(self,0,0,"AxisBrowserLayout")

        self.viewSetupFrame = QFrame(self,"viewSetupFrame")
        vsfsp = self.viewSetupFrame.sizePolicy()
        self.viewSetupFrame.setSizePolicy(QSizePolicy(QSizePolicy.Preferred,
                                                      QSizePolicy.Preferred,0,0,
                                                      vsfsp.hasHeightForWidth()))
        #self.viewSetupFrame.setMinimumSize(QSize(350,120))
        self.viewSetupFrame.setFrameShape(QFrame.StyledPanel)
        self.viewSetupFrame.setFrameShadow(QFrame.Raised)
        viewSetupFrameLayout = QVBoxLayout(self.viewSetupFrame,11,0,
                                           "viewSetupFrameLayout")
        

        self.viewModeFrame = QFrame(self.viewSetupFrame,"viewModeFrame")
        self.viewModeFrame.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.viewModeFrame.sizePolicy().hasHeightForWidth()))
        viewModeFrameLayout = QHBoxLayout(self.viewModeFrame,4,6,"viewModeFrameLayout")

        self.viewModeLabel = QLabel(self.viewModeFrame,"viewModeLabel")
        self.viewModeLabel.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.viewModeLabel.sizePolicy().hasHeightForWidth()))
        viewModeFrameLayout.addWidget(self.viewModeLabel)

        self.modeSelectGroup = QButtonGroup(self.viewModeFrame,"modeSelectGroup")
        self.modeSelectGroup.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.modeSelectGroup.sizePolicy().hasHeightForWidth()))
        self.modeSelectGroup.setFrameShape(QButtonGroup.NoFrame)
        self.modeSelectGroup.setFrameShadow(QButtonGroup.Sunken)
        self.modeSelectGroup.setLineWidth(1)
        self.modeSelectGroup.setProperty("selectedId",QVariant(1))
        self.modeSelectGroup.setColumnLayout(0,Qt.Vertical)
        self.modeSelectGroup.layout().setSpacing(0)
        self.modeSelectGroup.layout().setMargin(0)
        modeSelectGroupLayout = QHBoxLayout(self.modeSelectGroup.layout())
        modeSelectGroupLayout.setAlignment(Qt.AlignTop)

        self.radio1DMode = QRadioButton(self.modeSelectGroup,"radio1DMode")
        modeSelectGroupLayout.addWidget(self.radio1DMode)

        if self.cuboidView.cuboid.get_ndims() > 1:
            self.radio2DMode = QRadioButton(self.modeSelectGroup,"radio2DMode")
            self.radio2DMode.setText(self.__tr("2D"))
            modeSelectGroupLayout.addWidget(self.radio2DMode)

        if self.cuboidView.cuboid.get_ndims() > 2:
            self.radio3DMode = QRadioButton(self.modeSelectGroup,"radio3DMode")
            self.radio3DMode.setText("3D")
            modeSelectGroupLayout.addWidget(self.radio3DMode)

        #if self.mode == self.MODE_2D:
        nb_axes = len(self.cuboidView.current_axes)
        if nb_axes == 2:
            self.radio2DMode.setChecked(1)
        elif nb_axes == 1: #self.mode == self.MODE_1D:
            self.radio1DMode.setChecked(1)
        elif nb_axes == 3: #self.mode == self.MODE_3D:
            self.radio3DMode.setChecked(1)
        
        viewModeFrameLayout.addWidget(self.modeSelectGroup)
        viewSetupFrameLayout.addWidget(self.viewModeFrame)


        self.cropModeFrame = QFrame(self.viewSetupFrame,"cropModeFrame")
        self.cropModeFrame.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.cropModeFrame.sizePolicy().hasHeightForWidth()))
        self.cropModeFrame.setFrameShape(QFrame.NoFrame)
        self.cropModeFrame.setFrameShadow(QFrame.Plain)
        cropModeFrameLayout = QHBoxLayout(self.cropModeFrame,4,6,"cropModeFrameLayout")

        self.cropModeLabel = QLabel(self.cropModeFrame,"cropModeLabel")
        self.cropModeLabel.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.cropModeLabel.sizePolicy().hasHeightForWidth()))
        cropModeFrameLayout.addWidget(self.cropModeLabel)

        self.cropSelectGroup = QButtonGroup(self.cropModeFrame,"cropSelectGroup")
        self.cropSelectGroup.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.cropSelectGroup.sizePolicy().hasHeightForWidth()))
        self.cropSelectGroup.setFrameShape(QButtonGroup.NoFrame)
        self.cropSelectGroup.setFrameShadow(QButtonGroup.Sunken)
        self.cropSelectGroup.setLineWidth(1)
        self.cropSelectGroup.setProperty("selectedId",QVariant(1))
        self.cropSelectGroup.setColumnLayout(0,Qt.Vertical)
        self.cropSelectGroup.layout().setSpacing(0)
        self.cropSelectGroup.layout().setMargin(0)
        cropSelectGroupLayout = QHBoxLayout(self.cropSelectGroup.layout())
        cropSelectGroupLayout.setAlignment(Qt.AlignTop)


        self.radioCropOff = QRadioButton(self.cropSelectGroup,"radioCropOff")
        self.radioCropOff.setEnabled(1)
        self.radioCropOn = QRadioButton(self.cropSelectGroup,"radioCropOn")
        cropSelectGroupLayout.addWidget(self.radioCropOn)
        self.radioCropOn.setEnabled(1)
        
        # if self.cuboidViews.cropOn():
        #     self.radioCropOn.setChecked(1)
        # else:
        self.radioCropOff.setChecked(1)
        cropSelectGroupLayout.addWidget(self.radioCropOff)
        cropModeFrameLayout.addWidget(self.cropSelectGroup)
        viewSetupFrameLayout.addWidget(self.cropModeFrame)

            
        self.curAxesLabel = QLabel(self.viewSetupFrame,"curAxesLabel")
        self.curAxesLabel.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed,0,0,self.curAxesLabel.sizePolicy().hasHeightForWidth()))
        viewSetupFrameLayout.addWidget(self.curAxesLabel)


        vaxn = self.cuboidView.cuboid.axes_names
        all_axn = self.cuboidView.cuboid.axes_names
        vaxn = self.cuboidView.current_axes + \
            list(set(all_axn).difference(self.cuboidView.current_axes))
        print 'self.cuboidView.current_axes:', self.cuboidView.current_axes
        print 'all_axn:', all_axn

        if debug: print 'AxisBrowser.__init__: building axisViewSelector...'
        self.axisSelector = ExclusiveBoxSelector(vaxn, 3, self.viewSetupFrame,
                                                 "axisSelector")
        
        viewSetupFrameLayout.addWidget(self.axisSelector)
        AxisBrowserLayout.addWidget(self.viewSetupFrame)
        
        self.axisSelector.setShownNumber(len(self.cuboidView.current_axes))
        # if self.mode == self.MODE_1D:
        #     self.axisSelector.setShownNumber(1)
        # elif self.mode == self.MODE_2D:
        #     self.axisSelector.setShownNumber(2)
        # elif self.mode == self.MODE_3D:
        #     self.axisSelector.setShownNumber(3)
            
        #print 'v.gca:', self.cuboidViews[0].getCurrentAxesNames()
        self.axisSelector.setSelection(self.cuboidView.current_axes)


        self.AxisSlicersFrame = QFrame(self,"AxisSlicersFrame")
        self.AxisSlicersFrame.setSizePolicy(QSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred,0,0,self.AxisSlicersFrame.sizePolicy().hasHeightForWidth()))

        #self.AxisSlicersFrame.setMinimumSize(QSize(300,100))
        #self.AxisSlicersFrame.setSizePolicy(QSizePolicy(QSizePolicy.Minimum,
        #                                                QSizePolicy.Minimum,0,0,self.AxisSlicersFrame.sizePolicy().hasHeightForWidth()))

        self.AxisSlicersFrame.setFrameShape(QFrame.StyledPanel)
        self.AxisSlicersFrame.setFrameShadow(QFrame.Raised)
        AxisSlicersFrameLayout = QHBoxLayout(self.AxisSlicersFrame,1,1,
                                             "AxisSlicersFrameLayout")
        
        i = 0
        self.axisSlicers = {}
        if debug: print 'AxisBrowser.__init__: parsing axes names ...'
        for axn in self.cuboidView.cuboid.axes_names:
            if debug : print 'axn = ', axn
            d = self.cuboidView.cuboid.axes_domains[axn]
            axsl = AxisSlicer(axn, d, self.AxisSlicersFrame, "axisSlicer"+str(i))
            if axn in self.axisSelector.getSelections():
                axsl.setEnabled(False)
            AxisSlicersFrameLayout.addWidget(axsl)
            self.connect(axsl, PYSIGNAL("sliceChanged"),  self.sliceChanged)
            self.axisSlicers[axn] = axsl
            i += 1

        # if self.cuboidView.isMasked():
        #     axn = self.cuboidViews[0].getMaskName()
        #     d = self.cuboidViews[0].getMaskLabels()
        #     self.maskSlicer = AxisSlicer(axn, d, self.AxisSlicersFrame,
        #                                  "maskSlicer")
        #     self.axisSlicers[axn] = self.maskSlicer
        #     self.connect(self.maskSlicer, PYSIGNAL("sliceChanged"),
        #                  self.maskSliceChanged)
        #     AxisSlicersFrameLayout.addWidget(self.maskSlicer)
            
        AxisBrowserLayout.addWidget(self.AxisSlicersFrame)

        self.languageChange()

        self.resize(QSize(100,230).expandedTo(self.minimumSizeHint()))
        self.clearWState(Qt.WState_Polished)

        self.connect(self.modeSelectGroup,SIGNAL("clicked(int)"),
                     self.modeSelectGroup_clicked)
        self.connect(self.cropSelectGroup,SIGNAL("clicked(int)"),
                     self.cropSelectGroup_clicked)

        self.connect(self.axisSelector,
                     PYSIGNAL("selectionChanged"), self.axisSelectionChanged)
        self.connect(self.axisSelector,
                     PYSIGNAL("selectionSwapped"), self.axisSelectionSwapped)

    def getAxisSlicer(self, axn):
        return self.axisSlicers[axn]

    def getAxesNames(self):
        return self.axisSlicers.keys()

    def maskSliceChanged(self, axn, i):
        if debug: print 'xndarrayViewRenderer.maskSliceChanged ...'
        for cv in self.cuboidViews:
            cv.setMaskLabel(i)
        
        if self.cuboidViews[0].cropOn():
            self.emit(PYSIGNAL('cuboidViewChanged'),(axn,i))
        else:
            self.emit(PYSIGNAL('maskLabelChanged'),(axn,i))

    def axisSelectionChanged(self, newAxes):
        if debug:print '~~~## axisSelectionChanged(self, newAxes=%s): ...' \
                %str(newAxes)
        
        self.updateSlicersState()
        self.updatexndarrayView()

    def axisSelectionSwapped(self, previous, new):
        if debug:print '#### axisSelectionSwapped'
        #self.updateSlicersState()
        #self.updatexndarrayView()
        if len(previous) < 2 or (previous==new):
            return
        if len(previous) == 2:
            self.cuboidView.swapAxes(previous[0], previous[1])
        elif len(previous) == 3:
            temp = [a for a in previous]
            i = 0
            while (temp != new):
                if temp[i] != new[i]:
                    a1,a2 = temp[i],new[i]
                    temp[i] = a2
                    temp[temp.index(a2,i+1)] = a1
                    self.cuboidView.swapAxes(a1, a2)
                i += 1
        self.emit(PYSIGNAL('cuboidViewChanged'),())
    
        
    def sliceChanged(self, axn, i):
        if debug:print 'changing slice in cuboid view ...'
        self.cuboidView.setSlice(axn, i)
        if debug:print 'emitting cuboidViewChanged ...'
        self.emit(PYSIGNAL('cuboidViewChanged'),())

    def languageChange(self):
        self.setCaption(self.__tr("browser"))
        self.viewModeLabel.setText(self.__tr("View Mode:"))
        self.modeSelectGroup.setTitle(QString.null)
        self.radio1DMode.setText(self.__tr("1D"))
        self.curAxesLabel.setText(self.__tr("Current Axes:"))
        self.cropModeLabel.setText(self.__tr("Crop to mask:"))
        self.cropSelectGroup.setTitle(QString.null)
        self.radioCropOn.setText(self.__tr("On"))
        self.radioCropOff.setText(self.__tr("Off"))

    def updateSlicersState(self):
        if debug:print 'updateSlicersState:'

        currentAxes = self.axisSelector.getSelections()
        if debug:print '### new axes :', currentAxes 
        for axsl in self.axisSlicers.itervalues():
            if axsl.axisName in currentAxes:
                axsl.setEnabled(False)
            else:
                axsl.setEnabled(True)

    def updatexndarrayView(self):
        if debug:print '~~ cuboidView.updatexndarrayView : ~~'

        currentAxes = self.axisSelector.getSelections()
        if debug:print 'currentAxes :', currentAxes

        self.cuboidView.setView(currentAxes)
        
        for axsl in self.axisSlicers.itervalues():
            #if axsl.axisName not in currentAxes:
            if axsl.isEnabled():
                self.cuboidView.setSlice(axsl.axisName, axsl.value())
        # for cv in self.cuboidViews:
        #     if cv.isMasked():
        #         cv.setMaskLabel(self.maskSlicer.value())

        if debug:print 'at end of cuboidView.updatexndarrayView : ~~'
        if debug:print 'emitting cuboidViewChanged'
        self.emit(PYSIGNAL("cuboidViewChanged"),())

    def cropSelectGroup_clicked(self, flag):
        if debug : print '~~~~cropSelect:', flag
        if debug : print '~~~~current crop state :', self.cuboidView.cropOn()
        if self.cuboidViews[0].cropOn() == flag:
            return
        for cv in self.cuboidViews:
            cv.setCrop(flag)
        if debug : print '~~~~cropSelect clicked ... emit cuboidViewChanged!'
        self.emit(PYSIGNAL("cuboidViewChanged"),())
            
                    
    def modeSelectGroup_clicked(self, mode):
        if debug: print "~~~~~~ AxisBrowser.modeSelectGroup_clicked(int): "
        if debug: print 'mode clicked =', mode
        if debug: print 'current mode =', self.mode
        if self.mode == mode:
            return
        self.mode = mode

        if mode == self.MODE_1D:
            if debug: print ' switch to mode1D'
            self.axisSelector.setShownNumber(1)
        elif mode == self.MODE_2D:
            if debug: print ' switch to mode2D'
            self.axisSelector.setShownNumber(2)
        elif mode == self.MODE_3D:
            if debug: print ' switch to mode3D'
            self.axisSelector.setShownNumber(3)

        self.updateSlicersState()
        #self.updatexndarrayView()
        
    def closeEvent(self, event):
        self.emit(PYSIGNAL("windowClosed"),())
        QWidget.closeEvent(self, event)

        
    def __tr(self,s,c = None):
        return qApp.translate("AxisBrowser",s,c)


class xndarrayViewLister(QDialog):
    
    def __init__(self, views, parent = None,name = None,modal = 0,fl = 0):
        QDialog.__init__(self,parent,name,modal,fl)

        if not name:
            self.setName("xndarrayViewLister")


        xndarrayViewListerLayout = QVBoxLayout(self,11,6,"xndarrayViewListerLayout")

        self.viewButton = QPushButton(self,"viewButton")
        xndarrayViewListerLayout.addWidget(self.viewButton)

        self.listBox1 = QListBox(self,"listBox1")
        self.listBox1.setSelectionMode(QListBox.Extended)
        xndarrayViewListerLayout.addWidget(self.listBox1)

        self.languageChange()

        self.resize(QSize(175,98).expandedTo(self.minimumSizeHint()))
        self.clearWState(Qt.WState_Polished)
        self.views = views
        self.viewWinCounts = {}
        self.listBox1.clear()
        #self.idHolders = {}
        #for viewName in sorted(views.keys()):
        #    idHold = IndexHolder(viewName)
        #    self.idHolders[viewName] = idHold
        #    QObject.connect(idHold, PYSIGNAL("transferedSignal"),
        #                    self.purgeViewer)
        for viewName in sorted(views.keys()):
            self.listBox1.insertItem(viewName)         

        self.fRenderers = {}
        self.fBrowsers = {}
        self.axisToSlicerMap = {}
        
        self.connect(self.listBox1,SIGNAL("doubleClicked(QListBoxItem*)"),
                     self.listBox1_doubleClicked)

        self.connect(self.viewButton,SIGNAL("clicked()"),
                     self.viewSelectionPushed)


    def purgeViewer(self, idObj):
        self.fRenderers.pop(idObj)
        fr = self.fBrowsers.pop(idObj)
        self.unregisterBrowserAxes(fr)
        
    def languageChange(self):
        self.setCaption("Object Browser")
        self.viewButton.setText("view selected")

    def getViews(self, idObj):
        
        print 'idObj:', idObj
        if not self.views.has_key(idObj):
            #curViews = [self.views[i] for i in idObj]
            #self.views[idObj] = moldViews(curViews)
            return self.views[idObj[0]]
        else:
            return [v.copy() for v in self.views[idObj]]

    def makeNewViewWinLabel(self, idObj):
        """
        Make a new unique label for a new window
        and update window counts.
        """
        if not self.viewWinCounts.has_key(idObj):
            self.viewWinCounts[idObj] = 0
            return str(idObj)
        else:
            self.viewWinCounts[idObj] += 1
            return '%s(%d)' %(str(idObj),self.viewWinCounts[idObj])
        
        
    def viewSelected(self, selection):
        if debug: print 'viewSelected ... -> ', selection
        idObj = tuple(sorted(selection))
        if debug: print 'sorted selection:', selection
        viewWinLabel = self.makeNewViewWinLabel(idObj)
        vs = self.getViews(idObj)
        fRenderer = display.xndarrayViewRenderer(vs, idObj[0])
        fRenderer.setCaption(viewWinLabel+'[V]')
        self.fRenderers[viewWinLabel] = fRenderer

        if debug : print 'building AxisBrowser from view :'
        if debug :
            print vs.cuboid.descrip()
        
        fBrowser = AxisBrowser(vs)
        fBrowser.setCaption(viewWinLabel+'[B]')
        self.fBrowsers[viewWinLabel] = fBrowser
        
        QObject.connect(fBrowser, PYSIGNAL("cuboidViewChanged"),
                        fRenderer.updateView)
        QObject.connect(fBrowser, PYSIGNAL("maskLabelChanged"),
                        fRenderer.setMaskLabel)
        QObject.connect(fBrowser, PYSIGNAL("windowClosed"),
                        fRenderer.close)
        QObject.connect(fRenderer, PYSIGNAL("windowClosed"),
                        fBrowser.close)
        QObject.connect(fRenderer, PYSIGNAL("positionClicked"),
                        self.syncPosition)
        idHold = IndexHolder(viewWinLabel)
        QObject.connect(fRenderer, PYSIGNAL("windowClosed"),
                        idHold)
        QObject.connect(idHold, PYSIGNAL("transferedSignal"),
                        self.purgeViewer)
        
        axesToRegister = list(vs.cuboid.axes_names)
        # if vs[0].isMasked():
        #     axesToRegister.append(vs[0].getMaskName())
        self.registerBrowserAxes(axesToRegister, fBrowser)
        fBrowser.show()
        fRenderer.show()
        
##         if not self.fRenderers.has_key(idObj):
##             vs = self.getViews(idObj)
##             fRenderer = display.xndarrayViewRenderer(vs)
##             fRenderer.setCaption(str(idObj)+'[V]')
            
##             self.fRenderers[idObj] = fRenderer
##             fBrowser = AxisBrowser(vs)
##             fBrowser.setCaption(str(idObj)+'[B]')
##             self.fBrowsers[idObj] = fBrowser
##             QObject.connect(fBrowser, PYSIGNAL("cuboidViewChanged"),
##                             fRenderer.updateView)
##             QObject.connect(fBrowser, PYSIGNAL("maskLabelChanged"),
##                             fRenderer.setMaskLabel)
##             QObject.connect(fBrowser, PYSIGNAL("windowClosed"),
##                             fRenderer.close)
##             QObject.connect(fRenderer, PYSIGNAL("windowClosed"),
##                             fBrowser.close)
##             QObject.connect(fRenderer, PYSIGNAL("positionClicked"),
##                             self.syncPosition)
##             #QObject.connect(fRenderer, PYSIGNAL("windowClosed"),
##             #                self.idHolders[idObj])

##             axesToRegister = vs[0].getAxesNames()
##             if vs[0].isMasked():
##                 axesToRegister.append(vs[0].getMaskName())
##             self.registerBrowserAxes(axesToRegister, fBrowser)
##             fBrowser.show()
##             fRenderer.show()
##         else:
##             fRenderer  = self.fRenderers[idObj]
##             fBrowser = self.fBrowsers[idObj]
##             fBrowser.show()
##             fRenderer.show()
        

    def viewSelectionPushed(self):

        # get selected items:
        print 'Selected :', self.listBox1.selectedItem()
        print 'current item :', self.listBox1.currentItem()
        print 'current text :', self.listBox1.currentText()
        selection = [str(self.listBox1.text(i)) for i in
                     filter(self.listBox1.isSelected,
                            xrange(self.listBox1.count()))]
        if len(selection) == 0:
            return
        self.viewSelected(selection)
            
        
        
    def listBox1_doubleClicked(self,a0):
        if debug:print "xndarrayViewLister.listBox1_doubleClicked(QListBoxItem*)..."
        if debug:print "a0:", a0
        if debug:print '-> ', a0.text()
        idObj = str(a0.text())
        self.viewSelected([idObj])

    def syncPosition(self, axes, positions):
        if debug: 
            print "syncPosition ..." 
            print 'axes:', axes
            print 'positions:', positions
        for an,p in zip(axes, positions):
            relatedSlicers = self.axisToSlicerMap.get(an, None)
            if relatedSlicers is not None:
                for relatedSlicer in relatedSlicers:
                    if relatedSlicer is not None:
                        i = relatedSlicer.getSliceFromRealValue(p)
                        if i is not None:
                            relatedSlicer.recieveSliceChange(i)
        
    def registerBrowserAxes(self, axesNames, browser):

        for an in axesNames:
            curSlicer = browser.getAxisSlicer(an)
            if not self.axisToSlicerMap.has_key(an):
                self.axisToSlicerMap[an] = set()
            relatedSlicers = self.axisToSlicerMap[an]
            for relatedSlicer in relatedSlicers:
                if relatedSlicer != curSlicer:
                    self.connect(relatedSlicer, PYSIGNAL("declareSliceChange"),
                                 curSlicer.recieveSliceChange)
                    self.connect(curSlicer, PYSIGNAL("declareSliceChange"),
                                 relatedSlicer.recieveSliceChange)
            relatedSlicers.add(curSlicer)
        if debug:
            print 'end of registerBrowserAxes :'
            print 'axisToSlicerMap :', self.axisToSlicerMap
            
    def unregisterBrowserAxes(self, browser):
        axesNames = browser.getAxesNames()
        for an in axesNames:
            curSlicer = browser.getAxisSlicer(an)
            relatedSlicers = self.axisToSlicerMap[an]
            for relatedSlicer in relatedSlicers:
                if relatedSlicer != curSlicer:
                    self.disconnect(relatedSlicer, PYSIGNAL("declareSliceChange"),
                                   curSlicer.recieveSliceChange)
                    self.disconnect(curSlicer, PYSIGNAL("declareSliceChange"),
                                   relatedSlicer.recieveSliceChange)
            relatedSlicers.remove(curSlicer)
        if debug:
            print 'end of registerBrowserAxes :'
            print 'axisToSlicerMap :', self.axisToSlicerMap
            

