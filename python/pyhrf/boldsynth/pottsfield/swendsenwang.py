

import numpy as _np
import sys
import copy as CM

debug = 0

def walkCluster(i, links, labels, l, remainingIndexes):
    #print 'walk from ', i
    #cluster.add(i)
    if debug: print '-> treating ...', i
    labels[i] = l
    #visited[i] = True
    if len(links) == 0:
        #print 'no more links'
        return
    #print 'links to :', links[i]
    for e in links[i]:
        if e in remainingIndexes:
            remainingIndexes.remove(e)
            walkCluster(e, links, labels, l, remainingIndexes)


LINK_GRAPH_METHOD = 1
LINK_MAT_METHOD = 0
LINK_CLUST_METHOD = 2

def linkNodes(RefGraph, beta, GraphNodesLabels, GraphLinks, RefGrphNgbhPosi):
    
    for i in xrange(len(RefGraph)):
        for j in xrange(len(RefGraph[i])):
            if debug: print '** %d and %d' %(i,RefGraph[i][j])
            if RefGraph[i][j]>i:  # to avoid two comparisons
                                  # for each edge (non-oriented graph)
                if debug: print '-> treating ...'
                u=(int)(_np.exp(beta)*_np.random.rand()>1. and \
                        GraphNodesLabels[i]==GraphNodesLabels[RefGraph[i][j]])
                if u and debug : print '  link!' 
                GraphLinks[i][j]=u
                
                if RefGrphNgbhPosi==None:
                    for k in xrange(len(RefGraph[RefGraph[i][j]])):
                        if RefGraph[RefGraph[i][j]][k]==i:
                            GraphLinks[RefGraph[i][j]][k]=u
                else:
                    GraphLinks[RefGraph[i][j]][RefGrphNgbhPosi[i][j]]=u



def linkNodesSets(RefGraph, beta, GraphNodesLabels, links, weights=None):
    
    for i in xrange(len(RefGraph)):
        for j in xrange(len(RefGraph[i])):
            if debug: print '** %d and %d' %(i,RefGraph[i][j])
            if RefGraph[i][j]>i:  # to avoid two comparisons for each edge
                                  # (non-oriented graph)
                if debug: print '-> treating ...'
                if weights is None:
                    u = (_np.exp(beta) * (1-_np.random.rand()) >1. and \
                             GraphNodesLabels[i]==GraphNodesLabels[RefGraph[i][j]])
                else:
                    if 0:
                        print 'i:',i,'j:',j
                        print 'RefGraph:', len(RefGraph)
                        print 'RefGraph[i]:', len(RefGraph[i])
                        print 'RefGraph[i][j]:',RefGraph[i][j]
                        print 'weights:', len(weights)
                        print 'weights[i]:', len(weights[i])
                        print 'weights[i][j]:',weights[i][j]
                    u = (_np.exp(beta * weights[i][j]) * (1-_np.random.rand())>1. and \
                             GraphNodesLabels[i]==GraphNodesLabels[RefGraph[i][j]])
                if u:
                    if debug: print '  link!'
                    idxToLink = RefGraph[i][j]
                    links[idxToLink].add(i)
                    links[i].add(idxToLink)
    
def pickLabels(RefGraph, GraphLinks, GraphNodesLabels, NbLabels,
               TempVec, NextTempVec):
    
    for i in xrange(len(GraphNodesLabels)):
        GraphNodesLabels[i]=-1

    for i in xrange(len(RefGraph)):
        if GraphNodesLabels[i]==-1:
            #choice of a label for the connected region
            TempLabel=_np.floor(NbLabels*_np.random.rand())
            #label spread...
            #...init
            TempVec[0]=i
            TempVecSize=1
            GraphNodesLabels[i]=TempLabel

            #...label spread
            while TempVecSize>0:
                NextTempVecSize=0
                for tmp in xrange(TempVecSize):
                    CurntEntry=TempVec[tmp]
                    for CurntNghbdID in xrange(len(RefGraph[CurntEntry])):
                        if (GraphLinks[CurntEntry][CurntNghbdID]==1 and GraphNodesLabels[RefGraph[CurntEntry][CurntNghbdID]]==-1):
                            GraphNodesLabels[RefGraph[CurntEntry][CurntNghbdID]]=TempLabel
                            NextTempVec[NextTempVecSize]=RefGraph[CurntEntry][CurntNghbdID]
                            NextTempVecSize=NextTempVecSize+1

                TempVecSize=NextTempVecSize

                TempVec[0:TempVecSize]=NextTempVec[0:TempVecSize]

def set_cluster_labels(links, labels, nbClasses):
    
    visited = set([])
    for ni, nl in enumerate(links):
        if ni not in visited:
            visited.add(ni)
            l = _np.random.randint(nbClasses)
            labels[ni] = l
            queue = set(nl)
            while len(queue)>0 :
                k = queue.pop()
                if k not in visited:
                    labels[k] = l
                    visited.add(k)
                    queue.update(links[k])


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def SwendsenWangSampler_graph(RefGraph,GraphNodesLabels,beta,NbLabels,
                              GraphLinks=None,RefGrphNgbhPosi=None, method=1,
                              weights=None):
    """
    image sampling with Swendsen-Wang algorithm
    input:
        * RefGraph: 
            List which contains the connectivity graph. Each entry represents
            a node of the graph and contains the list of its neighbors entry
            location in the graph.
            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is 
            the 10th node.
            => There exists i such that RefGraph[10][i]=2
        * GraphNodesLabels: 
            list containing the nodes labels. The sampler aims to modify 
            its values in function of beta and NbLabels.
        * beta: 
            normalization constant
        * NbLabels: 
            number of labels (connected voxel pairs are considered if their
            labels are equal)
        * GraphLinks:
            Same shape as RefGraph. Each entry indicates if the link of the
            corresponding edge in RefGraph is considered 
            (if yes ...=1 else ...=0).
            This optional list is used as a temporary variable and will be
            modified ! Defining it makes the algorithm faster (no memory
            allocation).
        * RefGrphNgbhPosi:
            Same shape as RefGraph. RefGrphNgbhPosi[i][j] indicates for which
            k is the link to i in RefGraph[RefGraph[i][j]][k]
            This optional list is never modified.
    output:
        * GraphNodesLabels:
            resampled nodes labels. (not returned but modified)
    """
    
    #initializations...
    NodesNb=len(RefGraph)
    
    if GraphLinks==None and method == LINK_MAT_METHOD:
        GraphLinks=CptDefaultGraphLinks(RefGraph)
    
        
        
    #...vectors which will avoid the use of a recursive method for label spreading
    TempVec=_np.zeros(NodesNb,dtype=int)
    NextTempVec=_np.zeros(NodesNb,dtype=int)

    
    #links the voxels
    #print 'RefGraph :'
    #print RefGraph

    if method == LINK_MAT_METHOD:
        if RefGrphNgbhPosi==None:
            RefGrphNgbhPosi=CptRefGrphNgbhPosi(RefGraph)
        linkNodes(RefGraph, beta, GraphNodesLabels, GraphLinks, RefGrphNgbhPosi)
        pickLabels(RefGraph, GraphLinks, GraphNodesLabels, NbLabels,
                   TempVec, NextTempVec)
            
    if method == LINK_GRAPH_METHOD:
        links = range(NodesNb)
        for i in xrange(NodesNb):
            links[i] = set()
        linkNodesSets(RefGraph, beta, GraphNodesLabels, links,
                      weights)
        if debug:
            print 'links :'
            print links
        set_cluster_labels(links, GraphNodesLabels, NbLabels)
        #remainingSites = set(range(NodesNb))
        #while len(remainingSites)>0:
        #    l = _np.random.randint(NbLabels)
        #    i = remainingSites.pop()
        #    walkCluster(i, links, GraphNodesLabels, l, remainingSites)

            
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Cpt_U_graph(RefGraph,GraphNodesLabels,GraphWeight=None):
    """
    Computes an estimation of U(Graph)
  inputs:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
        * GraphNodesLabels: list containing the nodes labels.
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding
                            edge in RefGraph. If not defined the weights are set to 1.0.
    output:
        * U value 
    """
    if GraphWeight==None:
        GraphWeight=CptDefaultGraphWeight(RefGraph)
        
    U=0.
    for i in xrange(len(RefGraph)):
        for j in xrange(len(RefGraph[i])):
            if RefGraph[i][j]>i \
                    and GraphNodesLabels[i]==GraphNodesLabels[RefGraph[i][j]] :
                U += GraphWeight[i][j]
    
    return U


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Cpt_Vec_U_graph(RefGraph,beta,LabelsNb,SamplesNb,
                    GraphWeight=None,GraphNodesLabels=None,
                    GraphLinks=None,RefGrphNgbhPosi=None):
    """
    Computes a given number of U for fields generated according to a given normalization constant Beta.
    Swendsen-Wang sampling is used to generate fields.
    input:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
        * beta: normalization constant
        * LabelsNb: Labels number
        * SamplesNb: Samples number for the U estimations
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding
                       edge in RefGraph. If not defined the weights are set to 1.0.
        * GraphNodesLabels: Optional list containing the nodes labels. The sampler aims to modify its values in function of 
                            beta and NbLabels. At this level this variable is seen as temporary and will be modified. Defining 
                            it slightly increases the calculation times.
        * GraphLinks: Same shape as RefGraph. Each entry indicates if the link of the corresponding
                           edge in RefGraph is considered (if yes ...=1 else ...=0).
                            At this level this variable is seen as temporary and will be modified. Defining 
                            it slightly increases the calculation times.
        * RefGrphNgbhPosi: Same shape as RefGraph. RefGrphNgbhPosi[i][j] indicates for which k is the link to i in 
                           RefGraph[RefGraph[i][j]][k]. This optional list is never modified. 
    output:
        * VecU: Vector of size SamplesNb containing the U computations
    """
    #initialization
    if GraphWeight is None:
        GraphWeight=CptDefaultGraphWeight(RefGraph)
    
    if GraphNodesLabels==None:
        GraphNodesLabels=CptDefaultGraphNodesLabels(RefGraph)
    else:
        for i in xrange(len(GraphNodesLabels)):
            GraphNodesLabels[i]=0
    
    if GraphLinks==None:
        GraphLinks=CptDefaultGraphLinks(RefGraph)
    
    if RefGrphNgbhPosi==None:
        RefGrphNgbhPosi=CptRefGrphNgbhPosi(RefGraph)
        
    #all estimates of ImagLoc will then be significant in the expectation calculation (initial field is homogeneous)
    SwendsenWangSampler_graph(RefGraph,GraphNodesLabels,beta,LabelsNb,
                              GraphLinks=GraphLinks,RefGrphNgbhPosi=RefGrphNgbhPosi,
                              weights=GraphWeight)
    
    #estimation
    VecU=_np.zeros(SamplesNb)
    
    for i in xrange(SamplesNb):
        SwendsenWangSampler_graph(RefGraph,GraphNodesLabels,beta,
                                  LabelsNb,GraphLinks=GraphLinks,RefGrphNgbhPosi=RefGrphNgbhPosi,
                                  weights=GraphWeight)
        
        VecU[i]=Cpt_U_graph(RefGraph,GraphNodesLabels,GraphWeight=GraphWeight)
    
    return VecU



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def CptRefGrphNgbhPosi(RefGraph):
    """
    computes the critical list CptRefGrphNgbhPosi from RefGraph
    imput:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
    output:
        * RefGrphNgbhPosi: Same shape as RefGraph. RefGrphNgbhPosi[i][j] indicates for which k is the link to i in 
                           RefGraph[RefGraph[i][j]][k]. It makes algorithms which run through the graph much faster
                           since it avoids a critical loop.
    """
    
    RefGrphNgbhPosi=[]
    for i in xrange(len(RefGraph)):
        RefGrphNgbhPosi.append([])
        
    for CurntEntry in xrange(len(RefGraph)):
        for CurntNghbdID in xrange(len(RefGraph[CurntEntry])):
            CurntNghbd=RefGraph[CurntEntry][CurntNghbdID]
            for k in xrange(len(RefGraph[CurntNghbd])):
                if RefGraph[CurntNghbd][k]==CurntEntry:
                    RefGrphNgbhPosi[CurntEntry].append(k)
    
    return RefGrphNgbhPosi


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def CptDefaultGraphNodesLabels(RefGraph):
    """
    computes a default list GraphNodesLabels from RefGraph
    imput:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
    output:
        * GraphNodesLabels: List containing the nodes labels (consiedered for the computation of U). The sampler 
          aims to modify its values in function of beta and NbLabels.
    """
    
    GraphNodesLabels=[]
    for i in xrange(len(RefGraph)):
        GraphNodesLabels.append(0)
    
    return GraphNodesLabels

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def CptDefaultGraphLinks(RefGraph):
    """
    computes a default list GraphLinks from RefGraph
    imput:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
    output:
        * GraphLinks: Same shape as RefGraph. Each entry indicates whether the link of the corresponding
                      edge in RefGraph is considered or not in Swensdsen-Wang Sampling (1 -> yes / 0 -> no).
            """
    
    GraphLinks=[]
    for i in xrange(len(RefGraph)):
        GraphLinks.append([])
        for j in xrange(len(RefGraph[i])):
            GraphLinks[i].append(1)
    
    return GraphLinks

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def CptDefaultGraphWeight(RefGraph):
    """
    computes a default list GraphWeight from RefGraph. Each edge weight is set to 1.0.
    imput:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
    output:
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding edge in RefGraph
            """
    GraphWeight = range(len(RefGraph))
    for i in xrange(len(RefGraph)):
        GraphWeight[i] = [1.] * len(RefGraph[i])
    return GraphWeight

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def GraphBetaMix(RefGraph,GraphNodesLabels,beta=0.5,NbLabels=2,NbIt=5,weights=None):
    """
    Generate a partition in GraphNodesLabels with respect to beta.
    input:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
        * GraphNodesLabels: list containing the nodes labels.
        * beta: correlation factor for all conditions
        * NbLabels: number of labels in all site conditions
        * NbIt: number of sampling steps with SwendsenWangSampler_graph
    output:
        * GraphNodesLabels: sampled GraphNodesLabels (not returned but modified)
    """
    
    for i in xrange(NbIt):
        SwendsenWangSampler_graph(RefGraph,GraphNodesLabels,beta,NbLabels,weights)
        

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ImageToGraph(Image,Mask,LabelOI=1,ConnectivityType=6):
    """
    Computes the connectivity graph of an image under a 3D mask voxels.
    inputs:
        * Image: the 3D image (label field).
        * Mask: corresponding 3D mask.
        * LabelOI: Voxels of Mask containing the label LabelOI are those considered.
        * ConnectivityType: controles the connectivity considered in the graph.
                            ConnectivityType=26 : 26-connectivity
                            ConnectivityType=18 : 18-connectivity
                            ConnectivityType=6 : 6-connectivity
    outputs:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                                  ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                       => There exists i such that RefGraph[10][i]=2
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding
                            edge in RefGraph
        * GraphNodesCoord: Coordinates of each node in Mask
        * GraphNodesLabels: list containing the nodes labels.
    """
    
    #initialization
    NBZ=Image.shape[0]
    NBY=Image.shape[1]
    NBX=Image.shape[2]
    NBZb=Mask.shape[0]
    NBYb=Mask.shape[1]
    NBXb=Mask.shape[2]
    
    if (NBZ!=NBZb) or (NBY!=NBYb) or (NBX!=NBXb):
        print "!!! image and mask shapes do not correspond  !!!"
    
    #label the 1-voxels of MaskID
    MaskID=_np.int16(Mask.copy())
    ID=0
    for i in xrange(NBZ):
        for j in xrange(NBY):
            for k in xrange(NBX):
                if MaskID[i,j,k]!=LabelOI:
                    MaskID[i,j,k]=-1
                else:
                    MaskID[i,j,k]=ID
                    ID=ID+1
    
    #creates the Mask under its graph formulation
    RefGraph=[]
    GraphWeight=[]
    GraphNodesCoord=[]
    GraphNodesLabels=[]
    
    ID=0
    for i in xrange(NBZ):
        for j in xrange(NBY):
            for k in xrange(NBX):
                if MaskID[i,j,k]!=-1:
                    #tests the consistency with the previous loops
                    if MaskID[i,j,k]==ID:
                        ID=ID+1
                    else:
                        print 'consistency problem in the graph creation'
                    
                    #creation of the MaskID[i,j,k]'th entry in RefGraph
                    RefGraph.append([])
                    GraphWeight.append([])
                    GraphNodesCoord.append([k,j,i])
                    GraphNodesLabels.append(Image[i,j,k])
                    
                    if ConnectivityType==6 or ConnectivityType==18 or ConnectivityType==26:
                        if (i<NBZ-1):
                            if Mask[i+1,j,k]==1:
                                RefGraph[-1].append(MaskID[i+1,j,k])
                                GraphWeight[-1].append(1.)
                        if (i>0):
                            if Mask[i-1,j,k]==1:
                                RefGraph[-1].append(MaskID[i-1,j,k])
                                GraphWeight[-1].append(1.)
                        if (j<NBY-1):
                            if Mask[i,j+1,k]==1:
                                RefGraph[-1].append(MaskID[i,j+1,k])
                                GraphWeight[-1].append(1.)
                        if (j>0):
                            if Mask[i,j-1,k]==1:
                                RefGraph[-1].append(MaskID[i,j-1,k])
                                GraphWeight[-1].append(1.)
                        if (k<NBX-1):
                            if Mask[i,j,k+1]==1:
                                RefGraph[-1].append(MaskID[i,j,k+1])
                                GraphWeight[-1].append(1.)
                        if (k>0):
                            if Mask[i,j,k-1]==1:
                                RefGraph[-1].append(MaskID[i,j,k-1])
                                GraphWeight[-1].append(1.)
                        
                    if ConnectivityType==18 or ConnectivityType==26:
                        if (i<NBZ-1) and (j<NBY-1):
                            if Mask[i+1,j+1,k]==1:
                                RefGraph[-1].append(MaskID[i+1,j+1,k])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i<NBZ-1) and (j>0):
                            if Mask[i+1,j-1,k]==1:
                                RefGraph[-1].append(MaskID[i+1,j-1,k])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i<NBZ-1) and (k<NBX-1):
                            if Mask[i+1,j,k+1]==1:
                                RefGraph[-1].append(MaskID[i+1,j,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i<NBZ-1) and (k>0):
                            if Mask[i+1,j,k-1]==1:
                                RefGraph[-1].append(MaskID[i+1,j,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i>0) and (j<NBY-1):
                            if Mask[i-1,j+1,k]==1:
                                RefGraph[-1].append(MaskID[i-1,j+1,k])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i>0) and (j>0):
                            if Mask[i-1,j-1,k]==1:
                                RefGraph[-1].append(MaskID[i-1,j-1,k])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i>0) and (k<NBX-1):
                            if Mask[i-1,j,k+1]==1:
                                RefGraph[-1].append(MaskID[i-1,j,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i>0) and (k>0):
                            if Mask[i-1,j,k-1]==1:
                                RefGraph[-1].append(MaskID[i-1,j,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (j<NBY-1) and (k<NBX-1):
                            if Mask[i,j+1,k+1]==1:
                                RefGraph[-1].append(MaskID[i,j+1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (j<NBY-1) and (k>0):
                            if Mask[i,j+1,k-1]==1:
                                RefGraph[-1].append(MaskID[i,j+1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (j>0) and (k<NBX-1):
                            if Mask[i,j-1,k+1]==1:
                                RefGraph[-1].append(MaskID[i,j-1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (j>0) and (k>0):
                            if Mask[i,j-1,k-1]==1:
                                RefGraph[-1].append(MaskID[i,j-1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                    if ConnectivityType==26:
                        if (i<NBZ-1) and (j<NBY-1) and (k<NBX-1):
                            if Mask[i+1,j+1,k+1]==1:
                                RefGraph[-1].append(MaskID[i+1,j+1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i<NBZ-1) and (j<NBY-1) and (k>0):
                            if Mask[i+1,j+1,k-1]==1:
                                RefGraph[-1].append(MaskID[i+1,j+1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i<NBZ-1) and (j>0) and (k<NBX-1):
                            if Mask[i+1,j-1,k+1]==1:
                                RefGraph[-1].append(MaskID[i+1,j-1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i<NBZ-1) and (j>0) and (k>0):
                            if Mask[i+1,j-1,k-1]==1:
                                RefGraph[-1].append(MaskID[i+1,j-1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i>0) and (j<NBY-1) and (k<NBX-1):
                            if Mask[i-1,j+1,k+1]==1:
                                RefGraph[-1].append(MaskID[i-1,j+1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i>0) and (j<NBY-1) and (k>0):
                            if Mask[i-1,j+1,k-1]==1:
                                RefGraph[-1].append(MaskID[i-1,j+1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i>0) and (j>0) and (k<NBX-1):
                            if Mask[i-1,j-1,k+1]==1:
                                RefGraph[-1].append(MaskID[i-1,j-1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i>0) and (j>0) and (k>0):
                            if Mask[i-1,j-1,k-1]==1:
                                RefGraph[-1].append(MaskID[i-1,j-1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
    
    return RefGraph,GraphWeight,GraphNodesCoord,GraphNodesLabels


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def MaskToGraph(Mask,LabelOI=1,ConnectivityType=6):
    """
    Computes the connectivity graph of in 3D mask voxels.
    inputs:
        * Mask: 3D mask.
        * LabelOI: Voxels of Mask containing the label LabelOI are those considered.
        * ConnectivityType: controles the connectivity considered in the graph.
                            ConnectivityType=26 : 26-connectivity
                            ConnectivityType=18 : 18-connectivity
                            ConnectivityType=6 : 6-connectivity
    outputs:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                                  ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                       => There exists i such that RefGraph[10][i]=2
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding
                            edge in RefGraph
        * GraphNodesCoord: Coordinates of each node in Mask
    """
    
    #initialization
    NBZ=Mask.shape[0]
    NBY=Mask.shape[1]
    NBX=Mask.shape[2]
    
    
    #label the 1-voxels of MaskID
    MaskID=_np.int16(Mask.copy())
    ID=0
    for i in xrange(NBZ):
        for j in xrange(NBY):
            for k in xrange(NBX):
                if MaskID[i,j,k]!=LabelOI:
                    MaskID[i,j,k]=-1
                else:
                    MaskID[i,j,k]=ID
                    ID=ID+1
    
    #creates the Mask under its graph formulation
    RefGraph=[]
    GraphWeight=[]
    GraphNodesCoord=[]
    
    ID=0
    for i in xrange(NBZ):
        for j in xrange(NBY):
            for k in xrange(NBX):
                if MaskID[i,j,k]!=-1:
                    #tests the consistency with the previous loops
                    if MaskID[i,j,k]==ID:
                        ID=ID+1
                    else:
                        print 'consistency problem in the graph creation'
                    
                    #creation of the MaskID[i,j,k]'th entry in RefGraph
                    RefGraph.append([])
                    GraphWeight.append([])
                    GraphNodesCoord.append([k,j,i])
                    
                    if ConnectivityType==6 or ConnectivityType==18 or ConnectivityType==26:
                        if (i<NBZ-1):
                            if Mask[i+1,j,k]==1:
                                RefGraph[-1].append(MaskID[i+1,j,k])
                                GraphWeight[-1].append(1.)
                        if (i>0):
                            if Mask[i-1,j,k]==1:
                                RefGraph[-1].append(MaskID[i-1,j,k])
                                GraphWeight[-1].append(1.)
                        if (j<NBY-1):
                            if Mask[i,j+1,k]==1:
                                RefGraph[-1].append(MaskID[i,j+1,k])
                                GraphWeight[-1].append(1.)
                        if (j>0):
                            if Mask[i,j-1,k]==1:
                                RefGraph[-1].append(MaskID[i,j-1,k])
                                GraphWeight[-1].append(1.)
                        if (k<NBX-1):
                            if Mask[i,j,k+1]==1:
                                RefGraph[-1].append(MaskID[i,j,k+1])
                                GraphWeight[-1].append(1.)
                        if (k>0):
                            if Mask[i,j,k-1]==1:
                                RefGraph[-1].append(MaskID[i,j,k-1])
                                GraphWeight[-1].append(1.)
                        
                    if ConnectivityType==18 or ConnectivityType==26:
                        if (i<NBZ-1) and (j<NBY-1):
                            if Mask[i+1,j+1,k]==1:
                                RefGraph[-1].append(MaskID[i+1,j+1,k])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i<NBZ-1) and (j>0):
                            if Mask[i+1,j-1,k]==1:
                                RefGraph[-1].append(MaskID[i+1,j-1,k])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i<NBZ-1) and (k<NBX-1):
                            if Mask[i+1,j,k+1]==1:
                                RefGraph[-1].append(MaskID[i+1,j,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i<NBZ-1) and (k>0):
                            if Mask[i+1,j,k-1]==1:
                                RefGraph[-1].append(MaskID[i+1,j,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i>0) and (j<NBY-1):
                            if Mask[i-1,j+1,k]==1:
                                RefGraph[-1].append(MaskID[i-1,j+1,k])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i>0) and (j>0):
                            if Mask[i-1,j-1,k]==1:
                                RefGraph[-1].append(MaskID[i-1,j-1,k])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i>0) and (k<NBX-1):
                            if Mask[i-1,j,k+1]==1:
                                RefGraph[-1].append(MaskID[i-1,j,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (i>0) and (k>0):
                            if Mask[i-1,j,k-1]==1:
                                RefGraph[-1].append(MaskID[i-1,j,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (j<NBY-1) and (k<NBX-1):
                            if Mask[i,j+1,k+1]==1:
                                RefGraph[-1].append(MaskID[i,j+1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (j<NBY-1) and (k>0):
                            if Mask[i,j+1,k-1]==1:
                                RefGraph[-1].append(MaskID[i,j+1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (j>0) and (k<NBX-1):
                            if Mask[i,j-1,k+1]==1:
                                RefGraph[-1].append(MaskID[i,j-1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                        if (j>0) and (k>0):
                            if Mask[i,j-1,k-1]==1:
                                RefGraph[-1].append(MaskID[i,j-1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(2.))
                    if ConnectivityType==26:
                        if (i<NBZ-1) and (j<NBY-1) and (k<NBX-1):
                            if Mask[i+1,j+1,k+1]==1:
                                RefGraph[-1].append(MaskID[i+1,j+1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i<NBZ-1) and (j<NBY-1) and (k>0):
                            if Mask[i+1,j+1,k-1]==1:
                                RefGraph[-1].append(MaskID[i+1,j+1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i<NBZ-1) and (j>0) and (k<NBX-1):
                            if Mask[i+1,j-1,k+1]==1:
                                RefGraph[-1].append(MaskID[i+1,j-1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i<NBZ-1) and (j>0) and (k>0):
                            if Mask[i+1,j-1,k-1]==1:
                                RefGraph[-1].append(MaskID[i+1,j-1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i>0) and (j<NBY-1) and (k<NBX-1):
                            if Mask[i-1,j+1,k+1]==1:
                                RefGraph[-1].append(MaskID[i-1,j+1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i>0) and (j<NBY-1) and (k>0):
                            if Mask[i-1,j+1,k-1]==1:
                                RefGraph[-1].append(MaskID[i-1,j+1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i>0) and (j>0) and (k<NBX-1):
                            if Mask[i-1,j-1,k+1]==1:
                                RefGraph[-1].append(MaskID[i-1,j-1,k+1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
                        if (i>0) and (j>0) and (k>0):
                            if Mask[i-1,j-1,k-1]==1:
                                RefGraph[-1].append(MaskID[i-1,j-1,k-1])
                                GraphWeight[-1].append(1./_np.sqrt(3.))
    
    return RefGraph,GraphWeight,GraphNodesCoord



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def GraphToImage(GraphNodesCoord,GraphNodesLabels,NBZ,NBY,NBX):
    """
    Computes a 3D image from a connectivity graph.
    input:
        * GraphNodesCoord: Coordinates of each node in Mask
        * GraphNodesLabels: Nodes labels. For example, GraphNodesLabels[i] is the label of node i.
        * NBZ: image size on Z axis
        * NBY: image size on Y axis
        * NBX: image size on X axis
    output:
        * Image
    g"""
    
    Image=_np.zeros((NBZ,NBY,NBX),dtype=int)
    
    for i in xrange(len(GraphNodesCoord)):
        Image[GraphNodesCoord[i][2],GraphNodesCoord[i][1],GraphNodesCoord[i][0]]=GraphNodesLabels[i]
        
    return Image
