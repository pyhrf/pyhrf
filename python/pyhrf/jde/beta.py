# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import numpy as _np
import scipy.interpolate

from pyhrf.boldsynth.pottsfield.swendsenwang import *
from pyhrf.boldsynth.field import *

import pyhrf
from pyhrf.stats.random import truncRandn, erf

from pyhrf.tools import resampleToGrid, get_2Dtable_string

from pyhrf import xmlio
from pyhrf.xmlio.xmlnumpy import NumpyXMLHandler
from pyhrf.ndarray import xndarray
from samplerbase import *


if hasattr(_np, 'float96'):
    float_hires = np.float96
elif hasattr(_np, 'float128'):
    float_hires = np.float128
else:
    float_hires = np.float64

#################################
# Partition Function estimation #
#################################

def Cpt_Expected_U_graph(RefGraph,beta,LabelsNb,SamplesNb,GraphWeight=None,GraphNodesLabels=None,GraphLinks=None,RefGrphNgbhPosi=None):
    """
    Useless now!

    Estimates the expectation of U for a given normalization constant Beta and
    a given mask shape.
    Swendsen-Wang sampling is used to assess the expectation on significant
    images depending of beta.
    input:
        * RefGraph: List which contains the connectivity graph. Each entry
                    represents a node of the graph and contains the list of
                    its neighbors entry location in the graph.
                    ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd
                        node is the 10th node.
                    => There exists i such that RefGraph[10][i]=2
        * beta: normalization constant
        * LabelsNb: Labels number
        * SamplesNb: Samples number for the U expectation estimation
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the
                       corresponding edge in RefGraph. If not defined
                       the weights are set to 1.0.
        * GraphNodesLabels: Optional list containing the nodes labels.
                            The sampler aims to modify its values in function
                            of beta and NbLabels. At this level this variable
                            is seen as temporary and will be modified.
                            Defining it slightly increases the calculation
                            times.
        * GraphLinks: Same shape as RefGraph. Each entry indicates if the link
                      of the corresponding edge in RefGraph is considered
                      (if yes ...=1 else ...=0). At this level this variable
                      is seen as temporary and will be modified. Defining
                      it slightly increases the calculation times.
        * RefGrphNgbhPosi: Same shape as RefGraph. RefGrphNgbhPosi[i][j]
                           indicates for which k is the link to i in
                           RefGraph[RefGraph[i][j]][k]. This optional list
                           is never modified.
    output:
        * ExpectU: U expectation
    """
    #initialization
    SumU=0.

    if GraphWeight==None:
        GraphWeight=CptDefaultGraphWeight(RefGraph)

    if GraphNodesLabels==None:
        GraphNodesLabels=CptDefaultGraphNodesLabels(RefGraph)

    if GraphLinks==None:
        GraphLinks=CptDefaultGraphLinks(RefGraph)

    if RefGrphNgbhPosi==None:
        RefGrphNgbhPosi=CptRefGrphNgbhPosi(RefGraph)

    #all estimates of ImagLoc will then be significant in the expectation calculation
    for i in xrange(len(GraphNodesLabels)):
        GraphNodesLabels[i]=0

    SwendsenWangSampler_graph(RefGraph,GraphNodesLabels,beta,LabelsNb,
                              GraphLinks=GraphLinks,
                              RefGrphNgbhPosi=RefGrphNgbhPosi)


    #estimation
    for i in xrange(SamplesNb):
        SwendsenWangSampler_graph(RefGraph,GraphNodesLabels,beta,LabelsNb,
                                  GraphLinks=GraphLinks,
                                  RefGrphNgbhPosi=RefGrphNgbhPosi)
        Utemp=Cpt_U_graph(RefGraph,GraphNodesLabels,GraphWeight=GraphWeight)
        SumU=SumU+Utemp

    ExpectU=SumU/SamplesNb

    return ExpectU


def Estim_lnZ_ngbhd_graph(RefGraph,beta_Ngbhd,beta_Ref,lnZ_ref,VecU_ref,
                          LabelsNb):
    """
    Estimates ln(Z) for beta=betaNgbhd. beta_Ngbhd is supposed close to beta_Ref
    for which ln(Z) is known (lnZ_ref) and the energy U of fields generated
    according to it have already been
    computed (VecU_ref).
    input:
        * RefGraph: List which contains the connectivity graph. Each entry
                    represents a node of the graph and contains the list of
                    its neighbors entry location in the graph.
                    ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node
                    is the 10th node.
                    => There exists i such that RefGraph[10][i]=2
        * beta_Ngbhd: normalization constant for which ln(Z) will be estimated
        * beta_Ref: normalization constant close to beta_Ngbhd for which ln(Z)
                    already known
        * lnZ_ref: ln(Z) for beta=beta_Ref
        * VecU_ref: energy U of fields generated according to beta_Ref
        * LabelsNb: Labels number
    output:
        * lnZ_Ngbhd: ln(Z) for beta=beta_Ngbhd
    """

    #print 'VecU_ref:'
    #print VecU_ref
    LocSum = 0.

    #reference formulation
    #for i in xrange(VecU_ref.shape[0]):
    #    LocSum=LocSum+(np.exp(beta_Ngbhd*VecU_ref[i])/np.exp(beta_Ref*VecU_ref[i]))
    #LocSum=np.log(LocSum/VecU_ref.shape[0])

    #equivalent and numericaly more stable formulation
    for i in xrange(VecU_ref.shape[0]):
        LocSum = LocSum + (np.exp((beta_Ngbhd - beta_Ref) * VecU_ref[i] -
                                   np.log(VecU_ref.shape[0])))
    LocSum = np.log(LocSum)

    lnZ_Ngbhd = lnZ_ref + LocSum
    #print 'lnZ_Ngbhd:', lnZ_Ngbhd
    return lnZ_Ngbhd




def Cpt_Vec_Estim_lnZ_Graph_fast3(RefGraph,LabelsNb,MaxErrorAllowed=5,
                                  BetaMax=1.4,BetaStep=0.05):
    """
    Estimate ln(Z(beta)) of Potts fields. The default Beta grid is between 0. and 1.4 with
    a step of 0.05. Extrapolation algorithm is used. Fast estimates are only performed for
    Ising fields (2 labels). Reference partition functions were pre-computed on Ising fields
    designed on regular and non-regular grids. They all respect a 6-connectivity system.
    input:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
            and contains the list of its neighbors entry location in the graph.
            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.  => There exists i such that RefGraph[10][i]=2
        * LabelsNb: possible number of labels in each site of the graph
        * MaxErrorAllowed: maximum error allowed in the graph estimation (in percents).
        * BetaMax: Z(beta,mask) will be computed for beta between 0 and BetaMax. Maximum considered value is 1.4
        * BetaStep: gap between two considered values of beta. Actual gaps are not exactly those asked but very close.
    output:
        * Est_lnZ: Vector containing the ln(Z(beta)) estimates
        * V_Beta: Vector of the same size as VecExpectZ containing the corresponding beta value
    """

    #launch a more general algorithm if the inputs are not appropriate
    if (LabelsNb!=2 and LabelsNb!=3) or BetaMax>1.4:
        [Est_lnZ,V_Beta]=Cpt_Vec_Estim_lnZ_Graph(RefGraph,LabelsNb,SamplesNb=30,BetaMax=BetaMax,BetaStep=BetaStep,GraphWeight=None)
        return Est_lnZ,V_Beta

    #initialisation

    #...default returned values
    V_Beta=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4])
    Est_lnZ=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    #...load reference partition functions
    [BaseLogPartFctRef,V_Beta_Ref]=LoadBaseLogPartFctRef()

    NbSites=[len(RefGraph)]
    NbCliques=[sum( [len(nl) for nl in RefGraph] ) ]
    #...NbSites
    s=len(RefGraph)
    #...NbCliques
    NbCliquesTmp=0
    for j in xrange(len(RefGraph)):
        NbCliquesTmp=NbCliquesTmp+len(RefGraph[j])
    c=NbCliquesTmp/2
    NbCliques.append(c)
    #...StdVal Nb neighbors / Moy Nb neighbors
    StdValCliquesPerSiteTmp=0.

    nc = NbCliques[-1] + 0.
    ns = NbSites[-1] + 0.
    for j in xrange(len(RefGraph)):
        if ns==1: #HACK
            ns_1 = 1.
        else:
            ns_1 = ns-1.
        StdValCliquesPerSiteTmp = StdValCliquesPerSiteTmp \
            + ( (nc/ns-len(RefGraph[j])/2.)**2. ) / ns
        StdNgbhDivMoyNgbh = np.sqrt(StdValCliquesPerSiteTmp) \
            / ( nc/(ns_1) )

    #extrapolation algorithm
    Best_MaxError=10000000.
    logN2=np.log(2.)
    logN3=np.log(3.)

    if LabelsNb==2:
        for i in BaseLogPartFctRef.keys():
            if BaseLogPartFctRef[i]['NbLabels']==2:
                MaxError=np.abs((BaseLogPartFctRef[i]['NbSites']-1.)*((1.*c)/(1.*BaseLogPartFctRef[i]['NbCliques']))-(s-1.))*logN2  #error at beta=0
                MaxError=MaxError+(np.abs(BaseLogPartFctRef[i]['StdNgbhDivMoyNgbh']-StdNgbhDivMoyNgbh))  #penalty added to the error at zero to penalyze different homogeneities of the neighboroud  (a bareer function would be cleaner for the conversion in percents)
                MaxError=MaxError*100./(s*logN2)    #to have a percentage of error
                if MaxError<Best_MaxError:
                    Best_MaxError=MaxError
                    BestI=i

        if Best_MaxError<MaxErrorAllowed:
            Est_lnZ=((c*1.)/(BaseLogPartFctRef[BestI]['NbCliques']*1.))*BaseLogPartFctRef[BestI]['LogPF']+(1-(c*1.)/(BaseLogPartFctRef[BestI]['NbCliques']*1.))*logN2
            V_Beta=V_Beta_Ref.copy()
        else:
            pyhrf.verbose(1, 'LnZ: path sampling')
            [Est_lnZ,V_Beta] = Cpt_Vec_Estim_lnZ_Graph(RefGraph,LabelsNb,SamplesNb=30,BetaMax=BetaMax,BetaStep=BetaStep,GraphWeight=None)

    if LabelsNb==3:
        for i in BaseLogPartFctRef.keys():
            if BaseLogPartFctRef[i]['NbLabels']==3:
                MaxError=np.abs((BaseLogPartFctRef[i]['NbSites']-1.)*((1.*c)/(1.*BaseLogPartFctRef[i]['NbCliques']))-(s-1.))*logN3  #error at beta=0
                MaxError=MaxError+(np.abs(BaseLogPartFctRef[i]['StdNgbhDivMoyNgbh']-StdNgbhDivMoyNgbh))  #penalty added to the error at zero to penalyze different homogeneities of the neighboroud  (a bareer function would be cleaner for the conversion in percents)
                MaxError=MaxError*100./(s*logN3)    #to have a percentage of error
                if MaxError<Best_MaxError:
                    Best_MaxError=MaxError
                    BestI=i

        if Best_MaxError<MaxErrorAllowed:
            Est_lnZ=((c*1.)/(BaseLogPartFctRef[BestI]['NbCliques']*1.))*BaseLogPartFctRef[BestI]['LogPF']+(1-(c*1.)/(BaseLogPartFctRef[BestI]['NbCliques']*1.))*logN3
            V_Beta=V_Beta_Ref.copy()
    	else:
            pyhrf.verbose(1, 'LnZ: path sampling')
            [Est_lnZ,V_Beta] = Cpt_Vec_Estim_lnZ_Graph(RefGraph,LabelsNb,SamplesNb=30,BetaMax=BetaMax,BetaStep=BetaStep,GraphWeight=None)

	#reduction of the domain
        if (BetaMax<1.4):
            temp=0
            while V_Beta[temp]<BetaMax and temp<V_Beta.shape[0]-2:
                temp=temp+1
            V_Beta=V_Beta[:temp]
            Est_lnZ=Est_lnZ[:temp]

	#domain resampling
	if (abs(BetaStep-0.05)>0.0001):
            v_Beta_Resample=[]
            cpt=0.
            while cpt<BetaMax+0.0001:
                v_Beta_Resample.append(cpt)
                cpt=cpt+BetaStep
            Est_lnZ=resampleToGrid(np.array(V_Beta),np.array(Est_lnZ),np.array(v_Beta_Resample))
            V_Beta=v_Beta_Resample

    return Est_lnZ,V_Beta



def Cpt_Vec_Estim_lnZ_Graph(RefGraph,LabelsNb,SamplesNb=40,BetaMax=1.4,
                            BetaStep=0.05,GraphWeight=None):
    """
    Estimates ln(Z) for fields of a given size and Beta values between 0 and
    BetaMax. Estimates of ln(Z) are first computed on a coarse grid of Beta
    values. They are then computed and returned on a fine grid. No
    approximation using precomputed partition function is performed here.
    input:
        * RefGraph: List which contains the connectivity graph. Each entry
                    represents a node of the graph and contains the list of
                    its neighbors entry location in the graph.
                    ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node
                    is the 10th node.
                    => There exists i such that RefGraph[10][i]=2
        * LabelsNb: number of labels
        * SamplesNb: number of fields estimated for each beta
        * BetaMax: Z(beta,mask) will be computed for beta between 0 and BetaMax
        * BetaStep: gap between two considered values of beta (...in the fine
                    grid. This gap in the coarse grid is automatically fixed
                    and depends on the graph size.)
         * GraphWeight: Same shape as RefGraph. Each entry is the weight of
                        the corresponding edge in RefGraph. If not defined
                        the weights are set to 1.0.
    output:
        * VecEstim_lnZ: Vector containing the ln(Z(beta,mask)) estimates
        * VecBetaVal: Vector of the same size as VecExpectZ containing the
                      corresponding beta value
    """

    #initialization
    if GraphWeight is None:
        GraphWeight=CptDefaultGraphWeight(RefGraph)

    GraphNodesLabels=CptDefaultGraphNodesLabels(RefGraph)
    GraphLinks=CptDefaultGraphLinks(RefGraph)
    RefGrphNgbhPosi=CptRefGrphNgbhPosi(RefGraph)

    if LabelsNb==2:
        if len(RefGraph)<20:
            BetaStepCoarse=0.01
        elif len(RefGraph)<50:
            BetaStepCoarse=0.05
        else:
            BetaStepCoarse=0.1
    else: # 3 in particular
        if len(RefGraph)<20:
            BetaStepCoarse=0.005
        elif len(RefGraph)<50:
            BetaStepCoarse=0.025
        else:
            BetaStepCoarse=0.05

    #BetaStepCoarse = BetaStep

    BetaLoc=0.
    ListEstim_lnZ=[]
    ListBetaVal=[]
    VecU=[]

    ListEstim_lnZ.append(len(RefGraph)*np.log(LabelsNb))
    ListBetaVal.append(BetaLoc)

    #print 'RefGraph:', len(RefGraph)
    #print 'GraphWeight:', len(GraphWeight)

    #compute the Z(beta_i) at a coarse resolution
    while BetaLoc<BetaMax+0.000001:
        VecU.append(Cpt_Vec_U_graph(RefGraph,BetaLoc,LabelsNb,SamplesNb,
                                    GraphWeight=GraphWeight,
                                    GraphNodesLabels=GraphNodesLabels,
                                    GraphLinks=GraphLinks,
                                    RefGrphNgbhPosi=RefGrphNgbhPosi))
        BetaLoc=BetaLoc+BetaStepCoarse
        Estim_lnZ=Estim_lnZ_ngbhd_graph(RefGraph,BetaLoc,ListBetaVal[-1],
                                        ListEstim_lnZ[-1],VecU[-1],LabelsNb)
        ListEstim_lnZ.append(Estim_lnZ)
        ListBetaVal.append(BetaLoc)
        pyhrf.verbose(4, 'beta=%1.4f ->  ln(Z)=%1.4f' \
                          %(ListBetaVal[-1],ListEstim_lnZ[-1]))
    VecU.append(Cpt_Vec_U_graph(RefGraph,BetaLoc,LabelsNb,SamplesNb,
                                GraphWeight=GraphWeight,
                                GraphNodesLabels=GraphNodesLabels,
                                GraphLinks=GraphLinks,
                                RefGrphNgbhPosi=RefGrphNgbhPosi))

    #compute the Z(beta_j) at a fine resolution
    BetaLoc=0.
    ListEstim_lnZ_f=[]
    ListBetaVal_f=[]

    while BetaLoc < BetaMax + 0.000001:
        ListBetaVal_f.append(BetaLoc)
        i_cor = int(ListBetaVal_f[-1] / BetaStepCoarse)
        LEZnew = Estim_lnZ_ngbhd_graph(RefGraph, ListBetaVal_f[-1],
                                       ListBetaVal[i_cor], ListEstim_lnZ[i_cor],
                                       VecU[i_cor],LabelsNb) * \
                                       (ListBetaVal[i_cor+1] - ListBetaVal_f[-1])/BetaStepCoarse
        LEZnew = LEZnew + Estim_lnZ_ngbhd_graph(RefGraph,ListBetaVal_f[-1],
                                                ListBetaVal[i_cor+1],
                                                ListEstim_lnZ[i_cor+1],
                                                VecU[i_cor+1],LabelsNb) * \
                                                (-ListBetaVal[i_cor]+ListBetaVal_f[-1])/BetaStepCoarse
        ListEstim_lnZ_f.append(LEZnew)
        BetaLoc = BetaLoc + BetaStep

    #cast the lists into vectors
    VecEstim_lnZ=np.zeros(len(ListEstim_lnZ_f))
    VecBetaVal=np.zeros(len(ListBetaVal_f))

    for i in xrange(len(ListBetaVal_f)):
        VecBetaVal[i]=ListBetaVal_f[i]
        VecEstim_lnZ[i]=ListEstim_lnZ_f[i]

    return np.array(VecEstim_lnZ),np.array(VecBetaVal)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def LoadBaseLogPartFctRef():
    """
    output:
        * BaseLogPartFctRef: dictionnary that contains the data base of log-PF (first value = nb labels / second value = nb. sites / third value = nb. cliques)
        * V_Beta_Ref: Beta grid corresponding to the log-PF values in 'Est_lnZ_Ref'
    """

    BaseLogPartFctRef={}

    #non-regular grids: Graphes10_02.pickle

    BaseLogPartFctRef[0]={}
    BaseLogPartFctRef[0]['LogPF']=np.array([16.64,17.29,17.97,18.66,19.37,20.09,20.82,21.58,22.36,23.15,23.95,24.77,25.61,26.46,27.32,28.21,29.12,30.03,30.97,31.92,32.88,33.86,34.85,35.86,36.88,37.91,38.95,40.02,41.08,42.17])
    BaseLogPartFctRef[0]['NbLabels']=2
    BaseLogPartFctRef[0]['NbCliques']=26
    BaseLogPartFctRef[0]['NbSites']=24
    BaseLogPartFctRef[0]['StdNgbhDivMoyNgbh']=0.3970
    BaseLogPartFctRef[1]={}
    BaseLogPartFctRef[1]['LogPF']=np.array([17.33,18.22,19.12,20.05,21.01,21.98,22.98,24.00,25.04,26.11,27.21,28.33,29.49,30.67,31.88,33.11,34.38,35.69,37.02,38.38,39.78,41.19,42.64,44.11,45.61,47.13,48.68,50.25,51.83,53.44])
    BaseLogPartFctRef[1]['NbLabels']=2
    BaseLogPartFctRef[1]['NbCliques']=35
    BaseLogPartFctRef[1]['NbSites']=25
    BaseLogPartFctRef[1]['StdNgbhDivMoyNgbh']=0.4114
    BaseLogPartFctRef[2]={}
    BaseLogPartFctRef[2]['LogPF']=np.array([15.94,16.63,17.34,18.05,18.79,19.53,20.30,21.08,21.88,22.70,23.54,24.38,25.25,26.13,27.04,27.97,28.91,29.88,30.85,31.86,32.87,33.90,34.94,36.00,37.08,38.17,39.28,40.40,41.54,42.69])
    BaseLogPartFctRef[2]['NbLabels']=2
    BaseLogPartFctRef[2]['NbCliques']=27
    BaseLogPartFctRef[2]['NbSites']=23
    BaseLogPartFctRef[2]['StdNgbhDivMoyNgbh']=0.4093
    BaseLogPartFctRef[3]={}
    BaseLogPartFctRef[3]['LogPF']=np.array([19.41,20.47,21.56,22.67,23.81,24.99,26.18,27.41,28.67,29.95,31.28,32.64,34.03,35.46,36.93,38.45,39.99,41.58,43.21,44.89,46.61,48.37,50.16,51.98,53.82,55.69,57.60,59.50,61.44,63.39])
    BaseLogPartFctRef[3]['NbLabels']=2
    BaseLogPartFctRef[3]['NbCliques']=42
    BaseLogPartFctRef[3]['NbSites']=28
    BaseLogPartFctRef[3]['StdNgbhDivMoyNgbh']=0.3542
    BaseLogPartFctRef[4]={}
    BaseLogPartFctRef[4]['LogPF']=np.array([16.64,17.47,18.33,19.21,20.11,21.02,21.96,22.92,23.90,24.91,25.94,27.00,28.08,29.19,30.33,31.47,32.66,33.88,35.14,36.40,37.70,39.02,40.37,41.74,43.15,44.58,46.03,47.49,48.98,50.48])
    BaseLogPartFctRef[4]['NbLabels']=2
    BaseLogPartFctRef[4]['NbCliques']=33
    BaseLogPartFctRef[4]['NbSites']=24
    BaseLogPartFctRef[4]['StdNgbhDivMoyNgbh']=0.4527
    BaseLogPartFctRef[5]={}
    BaseLogPartFctRef[5]['LogPF']=np.array([9.01,9.42,9.84,10.26,10.69,11.14,11.59,12.06,12.54,13.03,13.53,14.04,14.56,15.09,15.63,16.18,16.75,17.33,17.91,18.51,19.12,19.74,20.38,21.01,21.66,22.32,22.99,23.67,24.36,25.06])
    BaseLogPartFctRef[5]['NbLabels']=2
    BaseLogPartFctRef[5]['NbCliques']=16
    BaseLogPartFctRef[5]['NbSites']=13
    BaseLogPartFctRef[5]['StdNgbhDivMoyNgbh']=0.3160
    BaseLogPartFctRef[6]={}
    BaseLogPartFctRef[6]['LogPF']=np.array([20.10,21.04,21.99,22.98,23.99,25.02,26.08,27.15,28.26,29.38,30.54,31.72,32.93,34.17,35.42,36.70,38.02,39.36,40.73,42.12,43.53,44.99,46.46,47.97,49.49,51.03,52.60,54.18,55.79,57.42])
    BaseLogPartFctRef[6]['NbLabels']=2
    BaseLogPartFctRef[6]['NbCliques']=37
    BaseLogPartFctRef[6]['NbSites']=29
    BaseLogPartFctRef[6]['StdNgbhDivMoyNgbh']=0.3663
    BaseLogPartFctRef[7]={}
    BaseLogPartFctRef[7]['LogPF']=np.array([17.33,18.11,18.91,19.73,20.58,21.44,22.32,23.22,24.15,25.10,26.07,27.06,28.07,29.09,30.14,31.22,32.31,33.43,34.58,35.75,36.94,38.14,39.37,40.62,41.90,43.19,44.50,45.83,47.16,48.51])
    BaseLogPartFctRef[7]['NbLabels']=2
    BaseLogPartFctRef[7]['NbCliques']=31
    BaseLogPartFctRef[7]['NbSites']=25
    BaseLogPartFctRef[7]['StdNgbhDivMoyNgbh']=0.4257
    BaseLogPartFctRef[8]={}
    BaseLogPartFctRef[8]['LogPF']=np.array([27.03,28.52,30.05,31.61,33.21,34.85,36.53,38.26,40.04,41.84,43.68,45.58,47.54,49.53,51.58,53.68,55.84,58.06,60.35,62.69,65.07,67.51,70.02,72.56,75.14,77.75,80.42,83.12,85.83,88.56])
    BaseLogPartFctRef[8]['NbLabels']=2
    BaseLogPartFctRef[8]['NbCliques']=59
    BaseLogPartFctRef[8]['NbSites']=39
    BaseLogPartFctRef[8]['StdNgbhDivMoyNgbh']=0.3682
    BaseLogPartFctRef[9]={}
    BaseLogPartFctRef[9]['LogPF']=np.array([16.64,17.47,18.33,19.20,20.10,21.02,21.95,22.92,23.91,24.92,25.95,27.00,28.08,29.19,30.32,31.48,32.66,33.88,35.12,36.38,37.68,39.00,40.35,41.73,43.12,44.53,45.97,47.43,48.90,50.39])
    BaseLogPartFctRef[9]['NbLabels']=2
    BaseLogPartFctRef[9]['NbCliques']=33
    BaseLogPartFctRef[9]['NbSites']=24
    BaseLogPartFctRef[9]['StdNgbhDivMoyNgbh']=0.3521
    BaseLogPartFctRef[10]={}
    BaseLogPartFctRef[10]['LogPF']=np.array([15.25,15.93,16.62,17.34,18.07,18.83,19.60,20.38,21.18,22.00,22.85,23.70,24.58,25.48,26.40,27.33,28.28,29.25,30.25,31.26,32.28,33.33,34.40,35.48,36.58,37.70,38.83,39.98,41.14,42.32])
    BaseLogPartFctRef[10]['NbLabels']=2
    BaseLogPartFctRef[10]['NbCliques']=27
    BaseLogPartFctRef[10]['NbSites']=22
    BaseLogPartFctRef[10]['StdNgbhDivMoyNgbh']=0.4344
    BaseLogPartFctRef[11]={}
    BaseLogPartFctRef[11]['LogPF']=np.array([11.09,11.59,12.11,12.65,13.19,13.74,14.31,14.90,15.49,16.10,16.73,17.36,18.01,18.67,19.36,20.05,20.77,21.49,22.23,22.98,23.74,24.52,25.33,26.14,26.97,27.81,28.66,29.52,30.39,31.27])
    BaseLogPartFctRef[11]['NbLabels']=2
    BaseLogPartFctRef[11]['NbCliques']=20
    BaseLogPartFctRef[11]['NbSites']=16
    BaseLogPartFctRef[11]['StdNgbhDivMoyNgbh']=0.3508
    BaseLogPartFctRef[12]={}
    BaseLogPartFctRef[12]['LogPF']=np.array([18.02,18.96,19.91,20.91,21.92,22.95,24.00,25.08,26.18,27.32,28.47,29.66,30.87,32.11,33.38,34.67,36.01,37.37,38.77,40.20,41.66,43.15,44.68,46.23,47.79,49.39,51.02,52.67,54.34,56.03])
    BaseLogPartFctRef[12]['NbLabels']=2
    BaseLogPartFctRef[12]['NbCliques']=37
    BaseLogPartFctRef[12]['NbSites']=26
    BaseLogPartFctRef[12]['StdNgbhDivMoyNgbh']=0.3712
    BaseLogPartFctRef[13]={}
    BaseLogPartFctRef[13]['LogPF']=np.array([22.87,23.89,24.93,25.99,27.08,28.20,29.34,30.51,31.70,32.91,34.16,35.43,36.72,38.03,39.39,40.77,42.17,43.60,45.05,46.55,48.05,49.59,51.16,52.76,54.38,56.02,57.68,59.37,61.08,62.81])
    BaseLogPartFctRef[13]['NbLabels']=2
    BaseLogPartFctRef[13]['NbCliques']=40
    BaseLogPartFctRef[13]['NbSites']=33
    BaseLogPartFctRef[13]['StdNgbhDivMoyNgbh']=0.4181
    BaseLogPartFctRef[14]={}
    BaseLogPartFctRef[14]['LogPF']=np.array([17.33,18.19,19.07,19.97,20.90,21.85,22.81,23.81,24.82,25.86,26.92,28.02,29.13,30.27,31.45,32.65,33.88,35.14,36.42,37.75,39.09,40.46,41.86,43.28,44.72,46.18,47.66,49.17,50.68,52.21])
    BaseLogPartFctRef[14]['NbLabels']=2
    BaseLogPartFctRef[14]['NbCliques']=34
    BaseLogPartFctRef[14]['NbSites']=25
    BaseLogPartFctRef[14]['StdNgbhDivMoyNgbh']=0.4057
    BaseLogPartFctRef[15]={}
    BaseLogPartFctRef[15]['LogPF']=np.array([15.94,16.75,17.58,18.43,19.30,20.19,21.10,22.04,22.99,23.97,24.97,25.99,27.04,28.10,29.20,30.31,31.46,32.63,33.83,35.06,36.31,37.59,38.90,40.22,41.57,42.95,44.34,45.75,47.19,48.63])
    BaseLogPartFctRef[15]['NbLabels']=2
    BaseLogPartFctRef[15]['NbCliques']=32
    BaseLogPartFctRef[15]['NbSites']=23
    BaseLogPartFctRef[15]['StdNgbhDivMoyNgbh']=0.3787
    BaseLogPartFctRef[16]={}
    BaseLogPartFctRef[16]['LogPF']=np.array([32.58,34.22,35.91,37.63,39.40,41.20,43.06,44.95,46.90,48.89,50.92,52.99,55.13,57.32,59.56,61.85,64.19,66.59,69.06,71.59,74.17,76.79,79.47,82.21,84.96,87.77,90.63,93.50,96.41,99.35])
    BaseLogPartFctRef[16]['NbLabels']=2
    BaseLogPartFctRef[16]['NbCliques']=65
    BaseLogPartFctRef[16]['NbSites']=47
    BaseLogPartFctRef[16]['StdNgbhDivMoyNgbh']=0.3945
    BaseLogPartFctRef[17]={}
    BaseLogPartFctRef[17]['LogPF']=np.array([21.49,22.62,23.79,24.99,26.21,27.46,28.75,30.07,31.40,32.78,34.19,35.64,37.11,38.63,40.18,41.78,43.42,45.11,46.85,48.62,50.44,52.30,54.20,56.13,58.09,60.09,62.11,64.14,66.20,68.28])
    BaseLogPartFctRef[17]['NbLabels']=2
    BaseLogPartFctRef[17]['NbCliques']=45
    BaseLogPartFctRef[17]['NbSites']=31
    BaseLogPartFctRef[17]['StdNgbhDivMoyNgbh']=0.4178
    BaseLogPartFctRef[18]={}
    BaseLogPartFctRef[18]['LogPF']=np.array([13.17,13.78,14.40,15.04,15.69,16.36,17.05,17.75,18.47,19.19,19.94,20.70,21.48,22.29,23.10,23.93,24.77,25.64,26.52,27.43,28.35,29.28,30.24,31.21,32.19,33.20,34.21,35.25,36.30,37.36])
    BaseLogPartFctRef[18]['NbLabels']=2
    BaseLogPartFctRef[18]['NbCliques']=24
    BaseLogPartFctRef[18]['NbSites']=19
    BaseLogPartFctRef[18]['StdNgbhDivMoyNgbh']=0.3724
    BaseLogPartFctRef[19]={}
    BaseLogPartFctRef[19]['LogPF']=np.array([13.86,14.45,15.04,15.66,16.29,16.92,17.57,18.24,18.93,19.63,20.34,21.07,21.82,22.59,23.36,24.15,24.96,25.79,26.63,27.48,28.36,29.25,30.15,31.06,31.99,32.93,33.89,34.86,35.84,36.83])
    BaseLogPartFctRef[19]['NbLabels']=2
    BaseLogPartFctRef[19]['NbCliques']=23
    BaseLogPartFctRef[19]['NbSites']=20
    BaseLogPartFctRef[19]['StdNgbhDivMoyNgbh']=0.3940
    BaseLogPartFctRef[20]={}
    BaseLogPartFctRef[20]['LogPF']=np.array([20.10,21.11,22.15,23.22,24.30,25.42,26.56,27.72,28.92,30.14,31.39,32.68,34.00,35.36,36.73,38.14,39.59,41.09,42.63,44.19,45.79,47.42,49.07,50.75,52.46,54.21,55.98,57.75,59.56,61.37])
    BaseLogPartFctRef[20]['NbLabels']=2
    BaseLogPartFctRef[20]['NbCliques']=40
    BaseLogPartFctRef[20]['NbSites']=29
    BaseLogPartFctRef[20]['StdNgbhDivMoyNgbh']=0.3970
    BaseLogPartFctRef[21]={}
    BaseLogPartFctRef[21]['LogPF']=np.array([16.64,17.34,18.07,18.82,19.58,20.36,21.16,21.97,22.80,23.65,24.52,25.41,26.32,27.24,28.18,29.15,30.13,31.12,32.13,33.17,34.23,35.31,36.39,37.50,38.63,39.76,40.92,42.10,43.29,44.49])
    BaseLogPartFctRef[21]['NbLabels']=2
    BaseLogPartFctRef[21]['NbCliques']=28
    BaseLogPartFctRef[21]['NbSites']=24
    BaseLogPartFctRef[21]['StdNgbhDivMoyNgbh']=0.3872
    BaseLogPartFctRef[22]={}
    BaseLogPartFctRef[22]['LogPF']=np.array([22.87,24.03,25.22,26.43,27.68,28.96,30.27,31.61,32.97,34.38,35.81,37.28,38.78,40.32,41.87,43.48,45.13,46.81,48.52,50.29,52.11,53.93,55.82,57.73,59.66,61.64,63.64,65.67,67.71,69.79])
    BaseLogPartFctRef[22]['NbLabels']=2
    BaseLogPartFctRef[22]['NbCliques']=46
    BaseLogPartFctRef[22]['NbSites']=33
    BaseLogPartFctRef[22]['StdNgbhDivMoyNgbh']=0.4085
    BaseLogPartFctRef[23]={}
    BaseLogPartFctRef[23]['LogPF']=np.array([13.86,14.52,15.19,15.88,16.59,17.31,18.05,18.81,19.59,20.38,21.19,22.02,22.87,23.74,24.63,25.53,26.46,27.40,28.38,29.37,30.38,31.41,32.46,33.52,34.60,35.70,36.82,37.95,39.09,40.24])
    BaseLogPartFctRef[23]['NbLabels']=2
    BaseLogPartFctRef[23]['NbCliques']=26
    BaseLogPartFctRef[23]['NbSites']=20
    BaseLogPartFctRef[23]['StdNgbhDivMoyNgbh']=0.3543
    BaseLogPartFctRef[24]={}
    BaseLogPartFctRef[24]['LogPF']=np.array([15.25,15.98,16.73,17.49,18.28,19.09,19.91,20.75,21.61,22.50,23.40,24.32,25.25,26.22,27.21,28.21,29.25,30.31,31.38,32.48,33.60,34.74,35.91,37.10,38.30,39.52,40.76,42.02,43.31,44.60])
    BaseLogPartFctRef[24]['NbLabels']=2
    BaseLogPartFctRef[24]['NbCliques']=29
    BaseLogPartFctRef[24]['NbSites']=22
    BaseLogPartFctRef[24]['StdNgbhDivMoyNgbh']=0.3709
    BaseLogPartFctRef[25]={}
    BaseLogPartFctRef[25]['LogPF']=np.array([30.50,31.97,33.47,35.02,36.60,38.20,39.86,41.55,43.28,45.04,46.84,48.68,50.57,52.50,54.46,56.47,58.51,60.60,62.75,64.94,67.18,69.45,71.76,74.12,76.51,78.93,81.40,83.92,86.46,89.03])
    BaseLogPartFctRef[25]['NbLabels']=2
    BaseLogPartFctRef[25]['NbCliques']=58
    BaseLogPartFctRef[25]['NbSites']=44
    BaseLogPartFctRef[25]['StdNgbhDivMoyNgbh']=0.3879
    BaseLogPartFctRef[26]={}
    BaseLogPartFctRef[26]['LogPF']=np.array([20.10,21.04,21.99,22.97,23.98,25.01,26.07,27.13,28.23,29.36,30.51,31.68,32.89,34.12,35.37,36.66,37.98,39.31,40.69,42.08,43.50,44.96,46.43,47.96,49.49,51.05,52.63,54.23,55.85,57.49])
    BaseLogPartFctRef[26]['NbLabels']=2
    BaseLogPartFctRef[26]['NbCliques']=37
    BaseLogPartFctRef[26]['NbSites']=29
    BaseLogPartFctRef[26]['StdNgbhDivMoyNgbh']=0.4167
    BaseLogPartFctRef[27]={}
    BaseLogPartFctRef[27]['LogPF']=np.array([14.56,15.16,15.78,16.42,17.07,17.74,18.43,19.12,19.83,20.56,21.30,22.06,22.83,23.62,24.42,25.25,26.09,26.94,27.81,28.69,29.58,30.50,31.42,32.36,33.31,34.27,35.26,36.25,37.25,38.27])
    BaseLogPartFctRef[27]['NbLabels']=2
    BaseLogPartFctRef[27]['NbCliques']=24
    BaseLogPartFctRef[27]['NbSites']=21
    BaseLogPartFctRef[27]['StdNgbhDivMoyNgbh']=0.3669
    BaseLogPartFctRef[28]={}
    BaseLogPartFctRef[28]['LogPF']=np.array([30.50,32.10,33.73,35.41,37.11,38.86,40.65,42.50,44.39,46.32,48.29,50.32,52.39,54.51,56.67,58.89,61.18,63.52,65.94,68.40,70.91,73.47,76.10,78.79,81.53,84.31,87.12,89.97,92.82,95.71])
    BaseLogPartFctRef[28]['NbLabels']=2
    BaseLogPartFctRef[28]['NbCliques']=63
    BaseLogPartFctRef[28]['NbSites']=44
    BaseLogPartFctRef[28]['StdNgbhDivMoyNgbh']=0.4089
    BaseLogPartFctRef[29]={}
    BaseLogPartFctRef[29]['LogPF']=np.array([20.79,21.70,22.64,23.60,24.58,25.59,26.62,27.67,28.74,29.83,30.95,32.11,33.28,34.46,35.68,36.92,38.20,39.50,40.82,42.17,43.54,44.93,46.36,47.80,49.26,50.74,52.24,53.77,55.31,56.88])
    BaseLogPartFctRef[29]['NbLabels']=2
    BaseLogPartFctRef[29]['NbCliques']=36
    BaseLogPartFctRef[29]['NbSites']=30
    BaseLogPartFctRef[29]['StdNgbhDivMoyNgbh']=0.3835


    #non-regular grids: Graphes10_04.pickle
    BaseLogPartFctRef[30]={}
    BaseLogPartFctRef[30]['LogPF']=np.array([487.3,527.2,568.2,610.1,653.1,697.1,742.2,788.6,836.3,885.4,936.2,989.5,1045.,1106.,1170.,1238.,1308.,1379.,1453.,1527.,1602.,1678.,1754.,1831.,1908.,1985.,2063.,2141.,2219.,2297.])
    BaseLogPartFctRef[30]['NbLabels']=2
    BaseLogPartFctRef[30]['NbCliques']=1578
    BaseLogPartFctRef[30]['NbSites']=703
    BaseLogPartFctRef[30]['StdNgbhDivMoyNgbh']=0.2508
    BaseLogPartFctRef[31]={}
    BaseLogPartFctRef[31]['LogPF']=np.array([463.0,500.5,539.0,578.3,618.7,660.0,702.4,745.9,790.7,836.8,884.5,934.4,987.0,1044.,1104.,1167.,1233.,1300.,1368.,1437.,1507.,1578.,1649.,1721.,1793.,1866.,1938.,2011.,2084.,2157.])
    BaseLogPartFctRef[31]['NbLabels']=2
    BaseLogPartFctRef[31]['NbCliques']=1481
    BaseLogPartFctRef[31]['NbSites']=668
    BaseLogPartFctRef[31]['StdNgbhDivMoyNgbh']=0.2677
    BaseLogPartFctRef[32]={}
    BaseLogPartFctRef[32]['LogPF']=np.array([310.5,333.2,356.4,380.2,404.5,429.4,454.9,481.1,508.0,535.6,564.1,593.4,623.9,655.7,688.8,723.7,760.2,798.1,837.2,877.4,918.3,959.8,1002.,1044.,1087.,1130.,1173.,1217.,1261.,1304.])
    BaseLogPartFctRef[32]['NbLabels']=2
    BaseLogPartFctRef[32]['NbCliques']=894
    BaseLogPartFctRef[32]['NbSites']=448
    BaseLogPartFctRef[32]['StdNgbhDivMoyNgbh']=0.2893
    BaseLogPartFctRef[33]={}
    BaseLogPartFctRef[33]['LogPF']=np.array([470.0,508.6,548.3,589.0,630.6,673.3,717.0,762.1,808.3,856.0,905.5,957.6,1013.,1072.,1135.,1201.,1269.,1339.,1410.,1482.,1554.,1628.,1702.,1776.,1850.,1925.,2000.,2076.,2151.,2227.])
    BaseLogPartFctRef[33]['NbLabels']=2
    BaseLogPartFctRef[33]['NbCliques']=1529
    BaseLogPartFctRef[33]['NbSites']=678
    BaseLogPartFctRef[33]['StdNgbhDivMoyNgbh']=0.2701
    BaseLogPartFctRef[34]={}
    BaseLogPartFctRef[34]['LogPF']=np.array([496.3,536.3,577.4,619.4,662.5,706.6,751.9,798.3,846.0,895.1,945.9,998.5,1054.,1113.,1175.,1242.,1312.,1383.,1456.,1530.,1606.,1681.,1758.,1835.,1912.,1990.,2067.,2145.,2224.,2302.])
    BaseLogPartFctRef[34]['NbLabels']=2
    BaseLogPartFctRef[34]['NbCliques']=1582
    BaseLogPartFctRef[34]['NbSites']=716
    BaseLogPartFctRef[34]['StdNgbhDivMoyNgbh']=0.2517
    BaseLogPartFctRef[35]={}
    BaseLogPartFctRef[35]['LogPF']=np.array([512.2,554.6,597.9,642.5,688.0,734.6,782.4,831.6,882.0,933.8,987.8,1044.,1103.,1166.,1234.,1305.,1379.,1456.,1534.,1613.,1692.,1773.,1854.,1936.,2018.,2100.,2183.,2265.,2348.,2431.])
    BaseLogPartFctRef[35]['NbLabels']=2
    BaseLogPartFctRef[35]['NbCliques']=1672
    BaseLogPartFctRef[35]['NbSites']=739
    BaseLogPartFctRef[35]['StdNgbhDivMoyNgbh']=0.2348
    BaseLogPartFctRef[36]={}
    BaseLogPartFctRef[36]['LogPF']=np.array([520.6,563.0,606.6,651.1,696.7,743.5,791.4,840.6,891.1,943.3,997.2,1053.,1112.,1175.,1243.,1315.,1389.,1465.,1542.,1621.,1701.,1781.,1862.,1944.,2026.,2108.,2191.,2273.,2356.,2439.])
    BaseLogPartFctRef[36]['NbLabels']=2
    BaseLogPartFctRef[36]['NbCliques']=1677
    BaseLogPartFctRef[36]['NbSites']=751
    BaseLogPartFctRef[36]['StdNgbhDivMoyNgbh']=0.2473
    BaseLogPartFctRef[37]={}
    BaseLogPartFctRef[37]['LogPF']=np.array([499.8,540.1,581.3,623.6,667.0,711.4,756.9,803.6,851.6,901.0,952.0,1005.,1061.,1120.,1183.,1250.,1320.,1392.,1465.,1539.,1615.,1691.,1768.,1845.,1923.,2001.,2079.,2157.,2236.,2315.])
    BaseLogPartFctRef[37]['NbLabels']=2
    BaseLogPartFctRef[37]['NbCliques']=1591
    BaseLogPartFctRef[37]['NbSites']=721
    BaseLogPartFctRef[37]['StdNgbhDivMoyNgbh']=0.2585
    BaseLogPartFctRef[38]={}
    BaseLogPartFctRef[38]['LogPF']=np.array([526.8,570.6,615.6,661.6,708.8,757.1,806.7,857.6,910.0,964.1,1020.,1079.,1141.,1208.,1280.,1355.,1432.,1511.,1592.,1674.,1756.,1839.,1923.,2008.,2093.,2178.,2263.,2348.,2434.,2520.])
    BaseLogPartFctRef[38]['NbLabels']=2
    BaseLogPartFctRef[38]['NbCliques']=1732
    BaseLogPartFctRef[38]['NbSites']=760
    BaseLogPartFctRef[38]['StdNgbhDivMoyNgbh']=0.2602
    BaseLogPartFctRef[39]={}
    BaseLogPartFctRef[39]['LogPF']=np.array([497.7,537.3,577.9,619.5,662.1,705.8,750.5,796.4,843.6,892.2,942.3,994.2,1049.,1106.,1168.,1233.,1301.,1371.,1443.,1516.,1590.,1665.,1741.,1816.,1893.,1969.,2046.,2123.,2201.,2278.])
    BaseLogPartFctRef[39]['NbLabels']=2
    BaseLogPartFctRef[39]['NbCliques']=1565
    BaseLogPartFctRef[39]['NbSites']=718
    BaseLogPartFctRef[39]['StdNgbhDivMoyNgbh']=0.2564
    BaseLogPartFctRef[40]={}
    BaseLogPartFctRef[40]['LogPF']=np.array([455.4,492.3,530.1,568.9,608.7,649.3,691.0,733.9,777.9,823.2,870.0,919.0,970.3,1025.,1084.,1146.,1210.,1276.,1344.,1412.,1481.,1551.,1622.,1692.,1764.,1835.,1907.,1979.,2051.,2123.])
    BaseLogPartFctRef[40]['NbLabels']=2
    BaseLogPartFctRef[40]['NbCliques']=1459
    BaseLogPartFctRef[40]['NbSites']=657
    BaseLogPartFctRef[40]['StdNgbhDivMoyNgbh']=0.2619
    BaseLogPartFctRef[41]={}
    BaseLogPartFctRef[41]['LogPF']=np.array([499.1,540.3,582.6,626.1,670.5,716.0,762.7,810.6,859.8,910.6,963.4,1018.,1076.,1139.,1205.,1276.,1348.,1423.,1498.,1575.,1653.,1732.,1811.,1890.,1970.,2050.,2131.,2211.,2292.,2373.])
    BaseLogPartFctRef[41]['NbLabels']=2
    BaseLogPartFctRef[41]['NbCliques']=1631
    BaseLogPartFctRef[41]['NbSites']=720
    BaseLogPartFctRef[41]['StdNgbhDivMoyNgbh']=0.2496
    BaseLogPartFctRef[42]={}
    BaseLogPartFctRef[42]['LogPF']=np.array([429.1,463.3,498.4,534.3,571.2,608.9,647.6,687.4,728.3,770.3,813.9,859.3,907.0,957.9,1012.,1069.,1128.,1188.,1250.,1313.,1377.,1441.,1506.,1571.,1637.,1703.,1770.,1836.,1903.,1970.])
    BaseLogPartFctRef[42]['NbLabels']=2
    BaseLogPartFctRef[42]['NbCliques']=1353
    BaseLogPartFctRef[42]['NbSites']=619
    BaseLogPartFctRef[42]['StdNgbhDivMoyNgbh']=0.2770
    BaseLogPartFctRef[43]={}
    BaseLogPartFctRef[43]['LogPF']=np.array([445.7,481.0,517.2,554.3,592.3,631.2,671.1,712.1,754.1,797.4,842.2,888.8,937.8,989.9,1046.,1104.,1164.,1227.,1290.,1355.,1421.,1487.,1554.,1621.,1689.,1757.,1825.,1894.,1963.,2031.])
    BaseLogPartFctRef[43]['NbLabels']=2
    BaseLogPartFctRef[43]['NbCliques']=1395
    BaseLogPartFctRef[43]['NbSites']=643
    BaseLogPartFctRef[43]['StdNgbhDivMoyNgbh']=0.2785
    BaseLogPartFctRef[44]={}
    BaseLogPartFctRef[44]['LogPF']=np.array([501.1,541.7,583.2,625.7,669.2,713.8,759.6,806.5,854.8,904.5,955.8,1009.,1065.,1125.,1190.,1258.,1328.,1401.,1475.,1550.,1626.,1702.,1779.,1857.,1935.,2013.,2092.,2171.,2250.,2329.])
    BaseLogPartFctRef[44]['NbLabels']=2
    BaseLogPartFctRef[44]['NbCliques']=1599
    BaseLogPartFctRef[44]['NbSites']=723
    BaseLogPartFctRef[44]['StdNgbhDivMoyNgbh']=0.2602
    BaseLogPartFctRef[45]={}
    BaseLogPartFctRef[45]['LogPF']=np.array([473.4,512.1,551.7,592.2,633.8,676.3,720.0,764.8,811.0,858.5,907.6,958.8,1013.,1071.,1133.,1199.,1266.,1335.,1406.,1477.,1550.,1623.,1697.,1771.,1845.,1920.,1995.,2070.,2146.,2221.])
    BaseLogPartFctRef[45]['NbLabels']=2
    BaseLogPartFctRef[45]['NbCliques']=1526
    BaseLogPartFctRef[45]['NbSites']=683
    BaseLogPartFctRef[45]['StdNgbhDivMoyNgbh']=0.2644
    BaseLogPartFctRef[46]={}
    BaseLogPartFctRef[46]['LogPF']=np.array([388.9,418.0,447.8,478.4,509.8,541.8,574.7,608.4,643.1,678.7,715.5,753.6,793.4,835.3,879.7,926.1,974.4,1025.,1076.,1129.,1182.,1236.,1291.,1346.,1401.,1457.,1513.,1569.,1626.,1682.])
    BaseLogPartFctRef[46]['NbLabels']=2
    BaseLogPartFctRef[46]['NbCliques']=1151
    BaseLogPartFctRef[46]['NbSites']=561
    BaseLogPartFctRef[46]['StdNgbhDivMoyNgbh']=0.3102
    BaseLogPartFctRef[47]={}
    BaseLogPartFctRef[47]['LogPF']=np.array([555.9,603.4,652.0,701.8,752.9,805.1,858.9,913.8,970.6,1029.,1090.,1154.,1222.,1296.,1375.,1457.,1541.,1628.,1716.,1805.,1895.,1985.,2076.,2168.,2260.,2352.,2445.,2538.,2630.,2723.])
    BaseLogPartFctRef[47]['NbLabels']=2
    BaseLogPartFctRef[47]['NbCliques']=1874
    BaseLogPartFctRef[47]['NbSites']=802
    BaseLogPartFctRef[47]['StdNgbhDivMoyNgbh']=0.2385
    BaseLogPartFctRef[48]={}
    BaseLogPartFctRef[48]['LogPF']=np.array([454.7,490.6,527.3,564.9,603.6,643.1,683.6,725.2,767.9,811.8,857.2,904.3,953.5,1005.,1061.,1119.,1180.,1243.,1308.,1374.,1441.,1508.,1576.,1645.,1714.,1783.,1852.,1922.,1992.,2062.])
    BaseLogPartFctRef[48]['NbLabels']=2
    BaseLogPartFctRef[48]['NbCliques']=1417
    BaseLogPartFctRef[48]['NbSites']=656
    BaseLogPartFctRef[48]['StdNgbhDivMoyNgbh']=0.2654
    BaseLogPartFctRef[49]={}
    BaseLogPartFctRef[49]['LogPF']=np.array([441.5,477.3,514.0,551.5,590.0,629.5,669.8,711.4,754.0,798.0,843.6,891.3,942.0,996.0,1054.,1114.,1177.,1241.,1306.,1373.,1440.,1507.,1575.,1644.,1712.,1781.,1851.,1920.,1990.,2059.])
    BaseLogPartFctRef[49]['NbLabels']=2
    BaseLogPartFctRef[49]['NbCliques']=1413
    BaseLogPartFctRef[49]['NbSites']=637
    BaseLogPartFctRef[49]['StdNgbhDivMoyNgbh']=0.2899
    BaseLogPartFctRef[50]={}
    BaseLogPartFctRef[50]['LogPF']=np.array([547.6,595.0,643.5,693.2,744.2,796.4,850.0,905.1,961.7,1020.,1081.,1145.,1214.,1289.,1369.,1451.,1536.,1623.,1711.,1801.,1891.,1982.,2073.,2165.,2257.,2350.,2442.,2535.,2628.,2721.])
    BaseLogPartFctRef[50]['NbLabels']=2
    BaseLogPartFctRef[50]['NbCliques']=1872
    BaseLogPartFctRef[50]['NbSites']=790
    BaseLogPartFctRef[50]['StdNgbhDivMoyNgbh']=0.2272
    BaseLogPartFctRef[51]={}
    BaseLogPartFctRef[51]['LogPF']=np.array([396.5,427.5,459.3,491.9,525.3,559.6,594.7,630.6,667.6,705.6,745.0,785.8,828.7,874.6,923.2,974.3,1027.,1081.,1137.,1194.,1251.,1309.,1368.,1427.,1486.,1546.,1606.,1666.,1727.,1787.])
    BaseLogPartFctRef[51]['NbLabels']=2
    BaseLogPartFctRef[51]['NbCliques']=1226
    BaseLogPartFctRef[51]['NbSites']=572
    BaseLogPartFctRef[51]['StdNgbhDivMoyNgbh']=0.2877
    BaseLogPartFctRef[52]={}
    BaseLogPartFctRef[52]['LogPF']=np.array([469.3,507.6,547.1,587.6,628.9,671.4,714.9,759.5,805.5,852.7,901.9,953.3,1007.,1066.,1129.,1194.,1262.,1331.,1402.,1473.,1545.,1618.,1691.,1765.,1839.,1914.,1988.,2063.,2138.,2213.])
    BaseLogPartFctRef[52]['NbLabels']=2
    BaseLogPartFctRef[52]['NbCliques']=1520
    BaseLogPartFctRef[52]['NbSites']=677
    BaseLogPartFctRef[52]['StdNgbhDivMoyNgbh']=0.2647
    BaseLogPartFctRef[53]={}
    BaseLogPartFctRef[53]['LogPF']=np.array([445.0,481.0,517.9,555.7,594.5,634.1,674.9,716.6,759.5,803.9,849.8,897.8,948.8,1004.,1061.,1122.,1185.,1249.,1314.,1381.,1448.,1516.,1584.,1653.,1722.,1792.,1861.,1931.,2001.,2071.])
    BaseLogPartFctRef[53]['NbLabels']=2
    BaseLogPartFctRef[53]['NbCliques']=1422
    BaseLogPartFctRef[53]['NbSites']=642
    BaseLogPartFctRef[53]['StdNgbhDivMoyNgbh']=0.2876
    BaseLogPartFctRef[54]={}
    BaseLogPartFctRef[54]['LogPF']=np.array([474.8,512.9,551.9,591.9,632.9,674.8,717.9,762.1,807.4,854.2,902.4,952.7,1005.,1061.,1121.,1184.,1250.,1317.,1387.,1457.,1528.,1600.,1673.,1746.,1819.,1893.,1967.,2041.,2115.,2190.])
    BaseLogPartFctRef[54]['NbLabels']=2
    BaseLogPartFctRef[54]['NbCliques']=1504
    BaseLogPartFctRef[54]['NbSites']=685
    BaseLogPartFctRef[54]['StdNgbhDivMoyNgbh']=0.2660
    BaseLogPartFctRef[55]={}
    BaseLogPartFctRef[55]['LogPF']=np.array([448.5,484.6,521.7,559.7,598.6,638.5,679.4,721.3,764.5,809.0,855.0,903.1,953.8,1008.,1065.,1125.,1188.,1252.,1318.,1385.,1452.,1521.,1589.,1659.,1728.,1798.,1869.,1939.,2010.,2080.])
    BaseLogPartFctRef[55]['NbLabels']=2
    BaseLogPartFctRef[55]['NbCliques']=1429
    BaseLogPartFctRef[55]['NbSites']=647
    BaseLogPartFctRef[55]['StdNgbhDivMoyNgbh']=0.2674
    BaseLogPartFctRef[56]={}
    BaseLogPartFctRef[56]['LogPF']=np.array([488.7,528.1,568.6,610.1,652.5,695.9,740.6,786.3,833.4,881.9,932.0,984.3,1039.,1098.,1161.,1227.,1295.,1366.,1438.,1511.,1584.,1659.,1734.,1810.,1886.,1962.,2039.,2115.,2192.,2269.])
    BaseLogPartFctRef[56]['NbLabels']=2
    BaseLogPartFctRef[56]['NbCliques']=1559
    BaseLogPartFctRef[56]['NbSites']=705
    BaseLogPartFctRef[56]['StdNgbhDivMoyNgbh']=0.2582
    BaseLogPartFctRef[57]={}
    BaseLogPartFctRef[57]['LogPF']=np.array([478.3,517.6,557.9,599.3,641.5,684.7,729.1,774.7,821.6,870.0,920.0,972.4,1028.,1087.,1151.,1217.,1286.,1357.,1429.,1502.,1575.,1650.,1725.,1800.,1876.,1952.,2029.,2105.,2182.,2259.])
    BaseLogPartFctRef[57]['NbLabels']=2
    BaseLogPartFctRef[57]['NbCliques']=1552
    BaseLogPartFctRef[57]['NbSites']=690
    BaseLogPartFctRef[57]['StdNgbhDivMoyNgbh']=0.2564
    BaseLogPartFctRef[58]={}
    BaseLogPartFctRef[58]['LogPF']=np.array([486.6,526.0,566.4,607.8,650.3,693.7,738.3,784.0,831.1,879.5,929.6,981.8,1037.,1096.,1159.,1225.,1293.,1364.,1436.,1509.,1583.,1658.,1733.,1808.,1884.,1960.,2037.,2114.,2190.,2267.])
    BaseLogPartFctRef[58]['NbLabels']=2
    BaseLogPartFctRef[58]['NbCliques']=1557
    BaseLogPartFctRef[58]['NbSites']=702
    BaseLogPartFctRef[58]['StdNgbhDivMoyNgbh']=0.2720
    BaseLogPartFctRef[59]={}
    BaseLogPartFctRef[59]['LogPF']=np.array([417.3,450.5,484.7,519.6,555.5,592.2,629.8,668.4,708.2,749.1,791.4,835.6,882.3,932.6,986.0,1042.,1099.,1158.,1219.,1280.,1342.,1404.,1467.,1530.,1594.,1658.,1723.,1787.,1852.,1917.])
    BaseLogPartFctRef[59]['NbLabels']=2
    BaseLogPartFctRef[59]['NbCliques']=1316
    BaseLogPartFctRef[59]['NbSites']=602
    BaseLogPartFctRef[59]['StdNgbhDivMoyNgbh']=0.2885


    #non-regular grids: Graphes10_03.pickle
    BaseLogPartFctRef[90]={}
    BaseLogPartFctRef[90]['LogPF']=np.array([59.61,62.87,66.21,69.64,73.14,76.75,80.43,84.19,88.05,92.00,96.03,100.2,104.4,108.7,113.2,117.8,122.5,127.3,132.2,137.2,142.4,147.7,153.1,158.6,164.2,169.9,175.6,181.5,187.4,193.3])
    BaseLogPartFctRef[90]['NbLabels']=2
    BaseLogPartFctRef[90]['NbCliques']=129
    BaseLogPartFctRef[90]['NbSites']=86
    BaseLogPartFctRef[90]['StdNgbhDivMoyNgbh']=0.3588
    BaseLogPartFctRef[91]={}
    BaseLogPartFctRef[91]['LogPF']=np.array([270.3,289.0,308.1,327.7,347.7,368.3,389.3,410.9,433.0,455.7,479.0,503.1,528.0,553.8,580.8,608.9,638.2,668.8,700.3,732.6,765.5,799.0,833.0,867.4,902.1,937.1,972.4,1008.,1043.,1079.])
    BaseLogPartFctRef[91]['NbLabels']=2
    BaseLogPartFctRef[91]['NbCliques']=737
    BaseLogPartFctRef[91]['NbSites']=390
    BaseLogPartFctRef[91]['StdNgbhDivMoyNgbh']=0.3378
    BaseLogPartFctRef[92]={}
    BaseLogPartFctRef[92]['LogPF']=np.array([71.39,75.52,79.74,84.08,88.51,93.04,97.70,102.5,107.3,112.3,117.4,122.7,128.1,133.7,139.4,145.2,151.3,157.5,164.0,170.5,177.3,184.1,191.1,198.3,205.6,212.9,220.4,227.9,235.5,243.2])
    BaseLogPartFctRef[92]['NbLabels']=2
    BaseLogPartFctRef[92]['NbCliques']=163
    BaseLogPartFctRef[92]['NbSites']=103
    BaseLogPartFctRef[92]['StdNgbhDivMoyNgbh']=0.4274
    BaseLogPartFctRef[93]={}
    BaseLogPartFctRef[93]['LogPF']=np.array([207.3,220.6,234.3,248.4,262.8,277.5,292.6,308.1,323.9,340.2,356.9,374.0,391.7,409.8,428.6,448.0,468.1,489.0,510.6,533.0,555.8,579.2,603.1,627.3,651.8,676.6,701.6,726.8,752.2,777.6])
    BaseLogPartFctRef[93]['NbLabels']=2
    BaseLogPartFctRef[93]['NbCliques']=529
    BaseLogPartFctRef[93]['NbSites']=299
    BaseLogPartFctRef[93]['StdNgbhDivMoyNgbh']=0.3502
    BaseLogPartFctRef[94]={}
    BaseLogPartFctRef[94]['LogPF']=np.array([94.27,99.47,104.8,110.2,115.8,121.5,127.4,133.4,139.5,145.8,152.2,158.8,165.5,172.5,179.5,186.8,194.3,202.0,209.9,218.0,226.2,234.7,243.3,252.1,261.1,270.2,279.4,288.7,298.1,307.6])
    BaseLogPartFctRef[94]['NbLabels']=2
    BaseLogPartFctRef[94]['NbCliques']=205
    BaseLogPartFctRef[94]['NbSites']=136
    BaseLogPartFctRef[94]['StdNgbhDivMoyNgbh']=0.3809
    BaseLogPartFctRef[95]={}
    BaseLogPartFctRef[95]['LogPF']=np.array([88.03,93.55,99.21,105.0,111.0,117.1,123.3,129.6,136.2,142.9,149.8,156.9,164.2,171.8,179.5,187.5,195.8,204.4,213.3,222.5,231.8,241.4,251.1,261.0,271.0,281.1,291.3,301.6,312.0,322.4])
    BaseLogPartFctRef[95]['NbLabels']=2
    BaseLogPartFctRef[95]['NbCliques']=218
    BaseLogPartFctRef[95]['NbSites']=127
    BaseLogPartFctRef[95]['StdNgbhDivMoyNgbh']=0.3673
    BaseLogPartFctRef[96]={}
    BaseLogPartFctRef[96]['LogPF']=np.array([117.1,124.4,131.8,139.4,147.1,155.1,163.2,171.5,180.1,188.8,197.8,207.0,216.4,226.1,236.1,246.5,257.2,268.1,279.5,291.2,303.3,315.6,328.2,341.1,354.1,367.3,380.6,394.1,407.6,421.2])
    BaseLogPartFctRef[96]['NbLabels']=2
    BaseLogPartFctRef[96]['NbCliques']=285
    BaseLogPartFctRef[96]['NbSites']=169
    BaseLogPartFctRef[96]['StdNgbhDivMoyNgbh']=0.3804
    BaseLogPartFctRef[97]={}
    BaseLogPartFctRef[97]['LogPF']=np.array([64.46,68.38,72.40,76.53,80.75,85.06,89.51,94.03,98.66,103.4,108.3,113.3,118.5,123.7,129.2,134.8,140.5,146.5,152.7,159.0,165.5,172.1,178.9,185.8,192.9,200.0,207.2,214.5,221.8,229.1])
    BaseLogPartFctRef[97]['NbLabels']=2
    BaseLogPartFctRef[97]['NbCliques']=155
    BaseLogPartFctRef[97]['NbSites']=93
    BaseLogPartFctRef[97]['StdNgbhDivMoyNgbh']=0.3436
    BaseLogPartFctRef[98]={}
    BaseLogPartFctRef[98]['LogPF']=np.array([94.96,100.3,105.8,111.4,117.1,123.0,129.0,135.2,141.5,147.9,154.6,161.3,168.3,175.5,182.8,190.4,198.2,206.3,214.5,222.9,231.6,240.5,249.6,258.7,268.1,277.5,287.1,296.8,306.5,316.3])
    BaseLogPartFctRef[98]['NbLabels']=2
    BaseLogPartFctRef[98]['NbCliques']=211
    BaseLogPartFctRef[98]['NbSites']=137
    BaseLogPartFctRef[98]['StdNgbhDivMoyNgbh']=0.4104
    BaseLogPartFctRef[99]={}
    BaseLogPartFctRef[99]['LogPF']=np.array([86.64,91.87,97.23,102.7,108.3,114.0,119.9,126.0,132.1,138.5,145.0,151.6,158.4,165.5,172.7,180.2,187.9,195.9,204.1,212.6,221.2,230.1,239.1,248.3,257.5,267.0,276.5,286.1,295.7,305.5])
    BaseLogPartFctRef[99]['NbLabels']=2
    BaseLogPartFctRef[99]['NbCliques']=206
    BaseLogPartFctRef[99]['NbSites']=125
    BaseLogPartFctRef[99]['StdNgbhDivMoyNgbh']=0.3692
    BaseLogPartFctRef[100]={}
    BaseLogPartFctRef[100]['LogPF']=np.array([94.96,101.0,107.1,113.5,119.9,126.6,133.4,140.3,147.5,154.8,162.2,169.9,177.8,185.9,194.3,203.0,212.0,221.3,230.9,240.8,251.0,261.4,272.0,282.8,293.7,304.8,316.0,327.3,338.7,350.1])
    BaseLogPartFctRef[100]['NbLabels']=2
    BaseLogPartFctRef[100]['NbCliques']=238
    BaseLogPartFctRef[100]['NbSites']=137
    BaseLogPartFctRef[100]['StdNgbhDivMoyNgbh']=0.3203
    BaseLogPartFctRef[101]={}
    BaseLogPartFctRef[101]['LogPF']=np.array([115.8,122.9,130.2,137.6,145.3,153.1,161.1,169.3,177.7,186.3,195.1,204.2,213.6,223.2,233.2,243.5,254.2,265.1,276.4,288.1,300.0,312.3,324.7,337.3,350.1,363.0,376.1,389.3,402.6,416.0])
    BaseLogPartFctRef[101]['NbLabels']=2
    BaseLogPartFctRef[101]['NbCliques']=281
    BaseLogPartFctRef[101]['NbSites']=167
    BaseLogPartFctRef[101]['StdNgbhDivMoyNgbh']=0.3747
    BaseLogPartFctRef[102]={}
    BaseLogPartFctRef[102]['LogPF']=np.array([160.1,170.5,181.1,192.0,203.1,214.5,226.2,238.1,250.4,262.9,275.8,289.1,302.7,316.8,331.4,346.7,362.4,378.6,395.5,412.9,430.7,448.9,467.4,486.2,505.1,524.3,543.7,563.1,582.6,602.3])
    BaseLogPartFctRef[102]['NbLabels']=2
    BaseLogPartFctRef[102]['NbCliques']=409
    BaseLogPartFctRef[102]['NbSites']=231
    BaseLogPartFctRef[102]['StdNgbhDivMoyNgbh']=0.3669
    BaseLogPartFctRef[103]={}
    BaseLogPartFctRef[103]['LogPF']=np.array([105.4,112.0,118.9,125.9,133.1,140.5,148.0,155.7,163.6,171.7,180.0,188.6,197.4,206.5,215.9,225.7,235.8,246.2,257.0,268.0,279.3,290.9,302.7,314.7,326.8,339.1,351.5,363.9,376.5,389.1])
    BaseLogPartFctRef[103]['NbLabels']=2
    BaseLogPartFctRef[103]['NbCliques']=264
    BaseLogPartFctRef[103]['NbSites']=152
    BaseLogPartFctRef[103]['StdNgbhDivMoyNgbh']=0.3455
    BaseLogPartFctRef[104]={}
    BaseLogPartFctRef[104]['LogPF']=np.array([90.80,96.28,101.9,107.6,113.5,119.5,125.7,132.0,138.5,145.1,151.9,158.9,166.1,173.4,181.0,188.8,196.9,205.2,213.8,222.7,231.8,241.1,250.6,260.3,270.1,280.1,290.1,300.2,310.4,320.6])
    BaseLogPartFctRef[104]['NbLabels']=2
    BaseLogPartFctRef[104]['NbCliques']=216
    BaseLogPartFctRef[104]['NbSites']=131
    BaseLogPartFctRef[104]['StdNgbhDivMoyNgbh']=0.3877
    BaseLogPartFctRef[105]={}
    BaseLogPartFctRef[105]['LogPF']=np.array([84.56,89.31,94.17,99.15,104.2,109.4,114.7,120.2,125.8,131.5,137.3,143.4,149.5,155.8,162.3,168.9,175.7,182.7,189.8,197.1,204.7,212.4,220.2,228.2,236.4,244.6,253.1,261.6,270.2,278.9])
    BaseLogPartFctRef[105]['NbLabels']=2
    BaseLogPartFctRef[105]['NbCliques']=187
    BaseLogPartFctRef[105]['NbSites']=122
    BaseLogPartFctRef[105]['StdNgbhDivMoyNgbh']=0.3629
    BaseLogPartFctRef[106]={}
    BaseLogPartFctRef[106]['LogPF']=np.array([53.37,56.03,58.75,61.54,64.39,67.32,70.31,73.36,76.49,79.70,82.98,86.34,89.76,93.27,96.83,100.5,104.2,108.0,112.0,116.0,120.1,124.2,128.5,132.8,137.1,141.6,146.1,150.7,155.3,160.0])
    BaseLogPartFctRef[106]['NbLabels']=2
    BaseLogPartFctRef[106]['NbCliques']=105
    BaseLogPartFctRef[106]['NbSites']=77
    BaseLogPartFctRef[106]['StdNgbhDivMoyNgbh']=0.3897
    BaseLogPartFctRef[107]={}
    BaseLogPartFctRef[107]['LogPF']=np.array([83.18,88.08,93.10,98.22,103.5,108.9,114.4,120.0,125.7,131.7,137.7,143.9,150.3,156.9,163.6,170.6,177.7,185.0,192.6,200.4,208.3,216.5,224.8,233.3,241.9,250.6,259.5,268.4,277.4,286.4])
    BaseLogPartFctRef[107]['NbLabels']=2
    BaseLogPartFctRef[107]['NbCliques']=193
    BaseLogPartFctRef[107]['NbSites']=120
    BaseLogPartFctRef[107]['StdNgbhDivMoyNgbh']=0.3673
    BaseLogPartFctRef[108]={}
    BaseLogPartFctRef[108]['LogPF']=np.array([135.9,144.3,153.0,161.8,170.9,180.2,189.7,199.5,209.5,219.8,230.3,241.1,252.3,263.7,275.5,287.7,300.3,313.3,326.7,340.6,354.7,369.1,383.9,398.9,414.1,429.5,445.1,460.8,476.6,492.6])
    BaseLogPartFctRef[108]['NbLabels']=2
    BaseLogPartFctRef[108]['NbCliques']=334
    BaseLogPartFctRef[108]['NbSites']=196
    BaseLogPartFctRef[108]['StdNgbhDivMoyNgbh']=0.3608
    BaseLogPartFctRef[109]={}
    BaseLogPartFctRef[109]['LogPF']=np.array([123.4,131.5,139.9,148.5,157.3,166.3,175.5,185.0,194.7,204.6,214.8,225.4,236.3,247.5,259.1,271.2,283.7,296.8,310.4,324.2,338.5,353.0,367.7,382.6,397.7,412.9,428.3,443.7,459.2,474.8])
    BaseLogPartFctRef[109]['NbLabels']=2
    BaseLogPartFctRef[109]['NbCliques']=323
    BaseLogPartFctRef[109]['NbSites']=178
    BaseLogPartFctRef[109]['StdNgbhDivMoyNgbh']=0.3482
    BaseLogPartFctRef[110]={}
    BaseLogPartFctRef[110]['LogPF']=np.array([64.46,67.91,71.46,75.12,78.83,82.65,86.53,90.54,94.63,98.82,103.1,107.5,112.0,116.7,121.4,126.3,131.4,136.5,141.8,147.3,152.8,158.5,164.3,170.1,176.1,182.1,188.2,194.4,200.7,206.9])
    BaseLogPartFctRef[110]['NbLabels']=2
    BaseLogPartFctRef[110]['NbCliques']=137
    BaseLogPartFctRef[110]['NbSites']=93
    BaseLogPartFctRef[110]['StdNgbhDivMoyNgbh']=0.4218
    BaseLogPartFctRef[111]={}
    BaseLogPartFctRef[111]['LogPF']=np.array([161.5,172.4,183.5,194.9,206.5,218.5,230.7,243.2,256.1,269.3,282.8,296.8,311.2,326.1,341.5,357.5,374.1,391.4,409.3,427.6,446.5,465.8,485.4,505.2,525.3,545.6,566.0,586.6,607.2,628.0])
    BaseLogPartFctRef[111]['NbLabels']=2
    BaseLogPartFctRef[111]['NbCliques']=428
    BaseLogPartFctRef[111]['NbSites']=233
    BaseLogPartFctRef[111]['StdNgbhDivMoyNgbh']=0.3430
    BaseLogPartFctRef[112]={}
    BaseLogPartFctRef[112]['LogPF']=np.array([63.77,67.94,72.22,76.58,81.09,85.70,90.42,95.26,100.2,105.3,110.5,115.9,121.5,127.2,133.2,139.4,146.0,152.7,159.7,166.8,174.1,181.5,189.0,196.7,204.4,212.2,220.0,227.9,235.8,243.7])
    BaseLogPartFctRef[112]['NbLabels']=2
    BaseLogPartFctRef[112]['NbCliques']=165
    BaseLogPartFctRef[112]['NbSites']=92
    BaseLogPartFctRef[112]['StdNgbhDivMoyNgbh']=0.3796
    BaseLogPartFctRef[113]={}
    BaseLogPartFctRef[113]['LogPF']=np.array([261.3,279.8,298.7,318.1,338.0,358.4,379.2,400.7,422.6,445.2,468.3,492.4,517.3,543.1,570.2,598.5,628.2,658.9,690.5,723.0,756.1,789.7,823.8,858.1,892.8,927.8,962.9,998.1,1034.,1069.])
    BaseLogPartFctRef[113]['NbLabels']=2
    BaseLogPartFctRef[113]['NbCliques']=730
    BaseLogPartFctRef[113]['NbSites']=377
    BaseLogPartFctRef[113]['StdNgbhDivMoyNgbh']=0.3418
    BaseLogPartFctRef[114]={}
    BaseLogPartFctRef[114]['LogPF']=np.array([94.96,100.3,105.8,111.4,117.2,123.1,129.1,135.3,141.7,148.1,154.8,161.6,168.6,175.7,183.1,190.6,198.4,206.4,214.6,223.0,231.7,240.5,249.6,258.8,268.1,277.5,287.1,296.8,306.6,316.4])
    BaseLogPartFctRef[114]['NbLabels']=2
    BaseLogPartFctRef[114]['NbCliques']=212
    BaseLogPartFctRef[114]['NbSites']=137
    BaseLogPartFctRef[114]['StdNgbhDivMoyNgbh']=0.3695
    BaseLogPartFctRef[115]={}
    BaseLogPartFctRef[115]['LogPF']=np.array([77.63,82.91,88.35,93.92,99.60,105.4,111.4,117.5,123.8,130.2,136.8,143.7,150.8,158.1,165.8,173.7,182.1,190.6,199.5,208.6,217.9,227.4,237.0,246.7,256.5,266.4,276.4,286.4,296.5,306.6])
    BaseLogPartFctRef[115]['NbLabels']=2
    BaseLogPartFctRef[115]['NbCliques']=209
    BaseLogPartFctRef[115]['NbSites']=112
    BaseLogPartFctRef[115]['StdNgbhDivMoyNgbh']=0.3290
    BaseLogPartFctRef[116]={}
    BaseLogPartFctRef[116]['LogPF']=np.array([110.9,117.5,124.3,131.2,138.2,145.5,152.9,160.5,168.3,176.3,184.5,192.9,201.5,210.3,219.4,228.7,238.4,248.3,258.5,268.9,279.7,290.7,302.0,313.5,325.2,337.0,349.0,361.1,373.3,385.6])
    BaseLogPartFctRef[116]['NbLabels']=2
    BaseLogPartFctRef[116]['NbCliques']=260
    BaseLogPartFctRef[116]['NbSites']=160
    BaseLogPartFctRef[116]['StdNgbhDivMoyNgbh']=0.3698
    BaseLogPartFctRef[117]={}
    BaseLogPartFctRef[117]['LogPF']=np.array([59.61,63.17,66.80,70.53,74.33,78.24,82.22,86.32,90.50,94.80,99.20,103.7,108.4,113.2,118.1,123.2,128.4,133.9,139.4,145.2,151.0,157.0,163.1,169.3,175.6,181.9,188.4,194.9,201.4,208.0])
    BaseLogPartFctRef[117]['NbLabels']=2
    BaseLogPartFctRef[117]['NbCliques']=140
    BaseLogPartFctRef[117]['NbSites']=86
    BaseLogPartFctRef[117]['StdNgbhDivMoyNgbh']=0.3851
    BaseLogPartFctRef[118]={}
    BaseLogPartFctRef[118]['LogPF']=np.array([141.4,150.6,160.1,169.8,179.7,189.8,200.2,210.9,221.8,233.0,244.5,256.4,268.6,281.2,294.3,307.9,322.1,336.7,351.9,367.5,383.4,399.7,416.2,432.9,449.8,466.8,484.0,501.4,518.8,536.3])
    BaseLogPartFctRef[118]['NbLabels']=2
    BaseLogPartFctRef[118]['NbCliques']=364
    BaseLogPartFctRef[118]['NbSites']=204
    BaseLogPartFctRef[118]['StdNgbhDivMoyNgbh']=0.3663
    BaseLogPartFctRef[119]={}
    BaseLogPartFctRef[119]['LogPF']=np.array([108.1,115.3,122.7,130.3,138.1,146.0,154.1,162.5,171.0,179.8,188.8,198.1,207.7,217.7,228.1,239.0,250.2,261.8,273.8,286.1,298.7,311.5,324.5,337.7,351.0,364.5,378.0,391.6,405.4,419.1])
    BaseLogPartFctRef[119]['NbLabels']=2
    BaseLogPartFctRef[119]['NbCliques']=285
    BaseLogPartFctRef[119]['NbSites']=156
    BaseLogPartFctRef[119]['StdNgbhDivMoyNgbh']=0.3677


    #non-regular grids: Graphes10_06.pickle
    BaseLogPartFctRef[120]={}
    BaseLogPartFctRef[120]['LogPF']=np.array([638.4,698.4,760.0,822.9,887.6,953.8,1022.,1092.,1164.,1239.,1319.,1406.,1501.,1602.,1708.,1817.,1928.,2041.,2155.,2270.,2386.,2503.,2620.,2737.,2854.,2972.,3090.,3208.,3326.,3444.])
    BaseLogPartFctRef[120]['NbLabels']=2
    BaseLogPartFctRef[120]['NbCliques']=2370
    BaseLogPartFctRef[120]['NbSites']=921
    BaseLogPartFctRef[120]['StdNgbhDivMoyNgbh']=0.1781
    BaseLogPartFctRef[121]={}
    BaseLogPartFctRef[121]['LogPF']=np.array([642.5,702.6,764.1,827.2,891.9,958.2,1026.,1096.,1168.,1243.,1323.,1409.,1503.,1604.,1710.,1818.,1929.,2042.,2156.,2272.,2388.,2504.,2621.,2739.,2856.,2974.,3092.,3210.,3328.,3446.])
    BaseLogPartFctRef[121]['NbLabels']=2
    BaseLogPartFctRef[121]['NbCliques']=2373
    BaseLogPartFctRef[121]['NbSites']=927
    BaseLogPartFctRef[121]['StdNgbhDivMoyNgbh']=0.1773
    BaseLogPartFctRef[122]={}
    BaseLogPartFctRef[122]['LogPF']=np.array([641.9,701.9,763.5,826.5,891.2,957.5,1026.,1096.,1168.,1243.,1322.,1409.,1504.,1605.,1710.,1819.,1930.,2043.,2157.,2272.,2388.,2505.,2622.,2739.,2857.,2975.,3093.,3211.,3329.,3448.])
    BaseLogPartFctRef[122]['NbLabels']=2
    BaseLogPartFctRef[122]['NbCliques']=2374
    BaseLogPartFctRef[122]['NbSites']=926
    BaseLogPartFctRef[122]['StdNgbhDivMoyNgbh']=0.1803
    BaseLogPartFctRef[123]={}
    BaseLogPartFctRef[123]['LogPF']=np.array([640.5,700.8,762.5,825.8,890.7,957.1,1025.,1096.,1168.,1244.,1324.,1411.,1507.,1609.,1715.,1824.,1936.,2050.,2165.,2280.,2397.,2514.,2631.,2749.,2867.,2985.,3104.,3222.,3341.,3459.])
    BaseLogPartFctRef[123]['NbLabels']=2
    BaseLogPartFctRef[123]['NbCliques']=2381
    BaseLogPartFctRef[123]['NbSites']=924
    BaseLogPartFctRef[123]['StdNgbhDivMoyNgbh']=0.1770
    BaseLogPartFctRef[124]={}
    BaseLogPartFctRef[124]['LogPF']=np.array([652.3,714.1,777.4,842.4,909.0,977.2,1047.,1119.,1194.,1271.,1353.,1442.,1541.,1646.,1755.,1868.,1983.,2100.,2218.,2336.,2456.,2576.,2697.,2818.,2939.,3060.,3182.,3303.,3425.,3547.])
    BaseLogPartFctRef[124]['NbLabels']=2
    BaseLogPartFctRef[124]['NbCliques']=2442
    BaseLogPartFctRef[124]['NbSites']=941
    BaseLogPartFctRef[124]['StdNgbhDivMoyNgbh']=0.1680
    BaseLogPartFctRef[125]={}
    BaseLogPartFctRef[125]['LogPF']=np.array([641.9,702.3,764.1,827.5,892.4,959.0,1027.,1098.,1170.,1246.,1326.,1413.,1509.,1611.,1717.,1826.,1938.,2051.,2166.,2282.,2399.,2516.,2633.,2751.,2869.,2987.,3106.,3224.,3343.,3462.])
    BaseLogPartFctRef[125]['NbLabels']=2
    BaseLogPartFctRef[125]['NbCliques']=2383
    BaseLogPartFctRef[125]['NbSites']=926
    BaseLogPartFctRef[125]['StdNgbhDivMoyNgbh']=0.1770
    BaseLogPartFctRef[126]={}
    BaseLogPartFctRef[126]['LogPF']=np.array([637.0,696.6,757.7,820.3,884.4,950.1,1018.,1087.,1159.,1233.,1312.,1397.,1491.,1592.,1696.,1805.,1915.,2027.,2140.,2255.,2370.,2485.,2602.,2718.,2835.,2951.,3068.,3185.,3303.,3420.])
    BaseLogPartFctRef[126]['NbLabels']=2
    BaseLogPartFctRef[126]['NbCliques']=2354
    BaseLogPartFctRef[126]['NbSites']=919
    BaseLogPartFctRef[126]['StdNgbhDivMoyNgbh']=0.1802
    BaseLogPartFctRef[127]={}
    BaseLogPartFctRef[127]['LogPF']=np.array([650.9,712.3,775.1,839.6,905.6,973.4,1043.,1114.,1188.,1265.,1346.,1435.,1533.,1637.,1745.,1857.,1971.,2086.,2203.,2321.,2439.,2558.,2678.,2798.,2918.,3039.,3159.,3280.,3401.,3521.])
    BaseLogPartFctRef[127]['NbLabels']=2
    BaseLogPartFctRef[127]['NbCliques']=2424
    BaseLogPartFctRef[127]['NbSites']=939
    BaseLogPartFctRef[127]['StdNgbhDivMoyNgbh']=0.1725
    BaseLogPartFctRef[128]={}
    BaseLogPartFctRef[128]['LogPF']=np.array([643.2,703.9,765.9,829.6,894.9,961.7,1030.,1101.,1174.,1250.,1331.,1419.,1515.,1618.,1725.,1835.,1947.,2062.,2177.,2294.,2411.,2528.,2647.,2765.,2884.,3003.,3122.,3241.,3360.,3480.])
    BaseLogPartFctRef[128]['NbLabels']=2
    BaseLogPartFctRef[128]['NbCliques']=2395
    BaseLogPartFctRef[128]['NbSites']=928
    BaseLogPartFctRef[128]['StdNgbhDivMoyNgbh']=0.1779
    BaseLogPartFctRef[129]={}
    BaseLogPartFctRef[129]['LogPF']=np.array([628.0,686.2,745.7,806.6,869.2,933.3,999.1,1067.,1137.,1209.,1286.,1368.,1459.,1556.,1658.,1762.,1869.,1978.,2088.,2199.,2311.,2424.,2537.,2651.,2764.,2878.,2992.,3107.,3221.,3335.])
    BaseLogPartFctRef[129]['NbLabels']=2
    BaseLogPartFctRef[129]['NbCliques']=2296
    BaseLogPartFctRef[129]['NbSites']=906
    BaseLogPartFctRef[129]['StdNgbhDivMoyNgbh']=0.1849
    BaseLogPartFctRef[130]={}
    BaseLogPartFctRef[130]['LogPF']=np.array([648.1,708.9,771.4,835.3,900.9,968.1,1037.,1108.,1181.,1257.,1338.,1425.,1522.,1624.,1732.,1842.,1955.,2070.,2186.,2303.,2421.,2539.,2658.,2776.,2896.,3015.,3135.,3254.,3374.,3494.])
    BaseLogPartFctRef[130]['NbLabels']=2
    BaseLogPartFctRef[130]['NbCliques']=2405
    BaseLogPartFctRef[130]['NbSites']=935
    BaseLogPartFctRef[130]['StdNgbhDivMoyNgbh']=0.1743
    BaseLogPartFctRef[131]={}
    BaseLogPartFctRef[131]['LogPF']=np.array([643.9,704.5,766.6,830.3,895.5,962.2,1031.,1101.,1174.,1250.,1330.,1419.,1514.,1616.,1723.,1833.,1945.,2059.,2175.,2291.,2408.,2525.,2643.,2762.,2880.,2999.,3118.,3237.,3356.,3476.])
    BaseLogPartFctRef[131]['NbLabels']=2
    BaseLogPartFctRef[131]['NbCliques']=2392
    BaseLogPartFctRef[131]['NbSites']=929
    BaseLogPartFctRef[131]['StdNgbhDivMoyNgbh']=0.1722
    BaseLogPartFctRef[132]={}
    BaseLogPartFctRef[132]['LogPF']=np.array([630.8,689.2,749.0,810.3,873.0,937.5,1004.,1072.,1142.,1215.,1291.,1375.,1466.,1563.,1665.,1770.,1878.,1987.,2098.,2210.,2322.,2435.,2549.,2663.,2777.,2891.,3006.,3120.,3235.,3350.])
    BaseLogPartFctRef[132]['NbLabels']=2
    BaseLogPartFctRef[132]['NbCliques']=2306
    BaseLogPartFctRef[132]['NbSites']=910
    BaseLogPartFctRef[132]['StdNgbhDivMoyNgbh']=0.1862
    BaseLogPartFctRef[133]={}
    BaseLogPartFctRef[133]['LogPF']=np.array([651.6,713.5,776.9,842.0,908.6,977.0,1047.,1119.,1194.,1271.,1354.,1444.,1543.,1649.,1758.,1871.,1987.,2104.,2222.,2341.,2461.,2581.,2702.,2823.,2944.,3066.,3187.,3309.,3431.,3553.])
    BaseLogPartFctRef[133]['NbLabels']=2
    BaseLogPartFctRef[133]['NbCliques']=2446
    BaseLogPartFctRef[133]['NbSites']=940
    BaseLogPartFctRef[133]['StdNgbhDivMoyNgbh']=0.1669
    BaseLogPartFctRef[134]={}
    BaseLogPartFctRef[134]['LogPF']=np.array([650.2,711.4,774.3,838.7,904.6,972.4,1042.,1113.,1187.,1264.,1345.,1434.,1532.,1635.,1743.,1855.,1969.,2084.,2201.,2319.,2437.,2557.,2676.,2796.,2916.,3036.,3157.,3278.,3398.,3519.])
    BaseLogPartFctRef[134]['NbLabels']=2
    BaseLogPartFctRef[134]['NbCliques']=2422
    BaseLogPartFctRef[134]['NbSites']=938
    BaseLogPartFctRef[134]['StdNgbhDivMoyNgbh']=0.1660
    BaseLogPartFctRef[135]={}
    BaseLogPartFctRef[135]['LogPF']=np.array([639.1,699.6,761.7,825.3,890.5,957.4,1026.,1096.,1169.,1245.,1326.,1414.,1511.,1614.,1721.,1831.,1943.,2057.,2173.,2289.,2406.,2524.,2642.,2760.,2879.,2998.,3117.,3236.,3355.,3474.])
    BaseLogPartFctRef[135]['NbLabels']=2
    BaseLogPartFctRef[135]['NbCliques']=2392
    BaseLogPartFctRef[135]['NbSites']=922
    BaseLogPartFctRef[135]['StdNgbhDivMoyNgbh']=0.1761
    BaseLogPartFctRef[136]={}
    BaseLogPartFctRef[136]['LogPF']=np.array([635.6,694.9,755.6,817.8,881.6,946.9,1014.,1083.,1154.,1228.,1307.,1391.,1484.,1583.,1687.,1795.,1904.,2015.,2128.,2242.,2356.,2471.,2586.,2702.,2817.,2934.,3050.,3166.,3283.,3399.])
    BaseLogPartFctRef[136]['NbLabels']=2
    BaseLogPartFctRef[136]['NbCliques']=2340
    BaseLogPartFctRef[136]['NbSites']=917
    BaseLogPartFctRef[136]['StdNgbhDivMoyNgbh']=0.1841
    BaseLogPartFctRef[137]={}
    BaseLogPartFctRef[137]['LogPF']=np.array([647.4,708.3,770.8,834.9,900.3,967.6,1037.,1108.,1181.,1257.,1338.,1427.,1523.,1627.,1734.,1845.,1958.,2073.,2189.,2306.,2423.,2542.,2660.,2779.,2899.,3018.,3138.,3257.,3377.,3497.])
    BaseLogPartFctRef[137]['NbLabels']=2
    BaseLogPartFctRef[137]['NbCliques']=2406
    BaseLogPartFctRef[137]['NbSites']=934
    BaseLogPartFctRef[137]['StdNgbhDivMoyNgbh']=0.1698
    BaseLogPartFctRef[138]={}
    BaseLogPartFctRef[138]['LogPF']=np.array([638.4,698.2,759.3,821.9,886.1,952.0,1020.,1089.,1161.,1235.,1314.,1400.,1494.,1594.,1698.,1806.,1917.,2029.,2143.,2257.,2373.,2489.,2605.,2722.,2838.,2956.,3073.,3190.,3308.,3425.])
    BaseLogPartFctRef[138]['NbLabels']=2
    BaseLogPartFctRef[138]['NbCliques']=2359
    BaseLogPartFctRef[138]['NbSites']=921
    BaseLogPartFctRef[138]['StdNgbhDivMoyNgbh']=0.1759
    BaseLogPartFctRef[139]={}
    BaseLogPartFctRef[139]['LogPF']=np.array([652.9,714.5,777.7,842.4,908.7,976.8,1047.,1119.,1193.,1270.,1352.,1442.,1540.,1644.,1753.,1865.,1979.,2096.,2213.,2332.,2451.,2571.,2691.,2812.,2932.,3053.,3174.,3296.,3417.,3539.])
    BaseLogPartFctRef[139]['NbLabels']=2
    BaseLogPartFctRef[139]['NbCliques']=2436
    BaseLogPartFctRef[139]['NbSites']=942
    BaseLogPartFctRef[139]['StdNgbhDivMoyNgbh']=0.1721
    BaseLogPartFctRef[140]={}
    BaseLogPartFctRef[140]['LogPF']=np.array([636.3,695.6,756.4,818.7,882.6,948.0,1015.,1084.,1156.,1230.,1308.,1393.,1486.,1586.,1690.,1797.,1907.,2018.,2131.,2245.,2359.,2474.,2590.,2705.,2821.,2938.,3054.,3171.,3287.,3404.])
    BaseLogPartFctRef[140]['NbLabels']=2
    BaseLogPartFctRef[140]['NbCliques']=2343
    BaseLogPartFctRef[140]['NbSites']=918
    BaseLogPartFctRef[140]['StdNgbhDivMoyNgbh']=0.1785
    BaseLogPartFctRef[141]={}
    BaseLogPartFctRef[141]['LogPF']=np.array([628.0,686.7,746.7,808.2,871.4,936.2,1003.,1071.,1141.,1215.,1292.,1377.,1470.,1568.,1671.,1777.,1885.,1995.,2107.,2219.,2332.,2446.,2560.,2675.,2789.,2904.,3019.,3135.,3250.,3365.])
    BaseLogPartFctRef[141]['NbLabels']=2
    BaseLogPartFctRef[141]['NbCliques']=2316
    BaseLogPartFctRef[141]['NbSites']=906
    BaseLogPartFctRef[141]['StdNgbhDivMoyNgbh']=0.1839
    BaseLogPartFctRef[142]={}
    BaseLogPartFctRef[142]['LogPF']=np.array([643.2,703.3,765.0,828.1,892.7,959.1,1027.,1097.,1170.,1245.,1324.,1411.,1506.,1607.,1712.,1821.,1932.,2045.,2160.,2275.,2391.,2508.,2625.,2742.,2860.,2978.,3096.,3214.,3332.,3450.])
    BaseLogPartFctRef[142]['NbLabels']=2
    BaseLogPartFctRef[142]['NbCliques']=2374
    BaseLogPartFctRef[142]['NbSites']=928
    BaseLogPartFctRef[142]['StdNgbhDivMoyNgbh']=0.1787
    BaseLogPartFctRef[143]={}
    BaseLogPartFctRef[143]['LogPF']=np.array([638.4,697.8,758.6,821.0,884.9,950.6,1018.,1087.,1159.,1233.,1311.,1396.,1489.,1588.,1693.,1800.,1910.,2022.,2135.,2249.,2363.,2479.,2594.,2710.,2827.,2943.,3060.,3177.,3294.,3411.])
    BaseLogPartFctRef[143]['NbLabels']=2
    BaseLogPartFctRef[143]['NbCliques']=2348
    BaseLogPartFctRef[143]['NbSites']=921
    BaseLogPartFctRef[143]['StdNgbhDivMoyNgbh']=0.1739
    BaseLogPartFctRef[144]={}
    BaseLogPartFctRef[144]['LogPF']=np.array([622.4,680.0,739.0,799.6,861.6,925.1,990.3,1057.,1126.,1198.,1274.,1356.,1446.,1542.,1642.,1746.,1852.,1960.,2069.,2179.,2290.,2401.,2513.,2626.,2738.,2851.,2964.,3077.,3190.,3303.])
    BaseLogPartFctRef[144]['NbLabels']=2
    BaseLogPartFctRef[144]['NbCliques']=2274
    BaseLogPartFctRef[144]['NbSites']=898
    BaseLogPartFctRef[144]['StdNgbhDivMoyNgbh']=0.1871
    BaseLogPartFctRef[145]={}
    BaseLogPartFctRef[145]['LogPF']=np.array([642.5,702.9,764.9,828.3,893.3,959.9,1028.,1099.,1171.,1247.,1327.,1414.,1509.,1611.,1717.,1827.,1939.,2053.,2168.,2284.,2401.,2518.,2636.,2754.,2873.,2991.,3110.,3229.,3348.,3467.])
    BaseLogPartFctRef[145]['NbLabels']=2
    BaseLogPartFctRef[145]['NbCliques']=2387
    BaseLogPartFctRef[145]['NbSites']=927
    BaseLogPartFctRef[145]['StdNgbhDivMoyNgbh']=0.1717
    BaseLogPartFctRef[146]={}
    BaseLogPartFctRef[146]['LogPF']=np.array([619.0,675.5,733.5,792.9,853.7,916.0,980.1,1046.,1114.,1184.,1257.,1336.,1423.,1516.,1615.,1716.,1820.,1926.,2033.,2141.,2250.,2359.,2469.,2579.,2689.,2800.,2911.,3022.,3133.,3244.])
    BaseLogPartFctRef[146]['NbLabels']=2
    BaseLogPartFctRef[146]['NbCliques']=2233
    BaseLogPartFctRef[146]['NbSites']=893
    BaseLogPartFctRef[146]['StdNgbhDivMoyNgbh']=0.1868
    BaseLogPartFctRef[147]={}
    BaseLogPartFctRef[147]['LogPF']=np.array([642.5,702.4,763.8,826.8,891.4,957.7,1026.,1095.,1167.,1242.,1321.,1407.,1502.,1602.,1707.,1816.,1927.,2039.,2153.,2269.,2384.,2501.,2617.,2734.,2852.,2969.,3087.,3205.,3323.,3441.])
    BaseLogPartFctRef[147]['NbLabels']=2
    BaseLogPartFctRef[147]['NbCliques']=2369
    BaseLogPartFctRef[147]['NbSites']=927
    BaseLogPartFctRef[147]['StdNgbhDivMoyNgbh']=0.1785
    BaseLogPartFctRef[148]={}
    BaseLogPartFctRef[148]['LogPF']=np.array([637.0,696.7,757.7,820.3,884.5,950.3,1018.,1087.,1159.,1233.,1312.,1398.,1492.,1592.,1697.,1805.,1915.,2027.,2141.,2255.,2370.,2486.,2602.,2718.,2835.,2952.,3069.,3186.,3303.,3421.])
    BaseLogPartFctRef[148]['NbLabels']=2
    BaseLogPartFctRef[148]['NbCliques']=2355
    BaseLogPartFctRef[148]['NbSites']=919
    BaseLogPartFctRef[148]['StdNgbhDivMoyNgbh']=0.1801
    BaseLogPartFctRef[149]={}
    BaseLogPartFctRef[149]['LogPF']=np.array([643.9,704.3,766.1,829.5,894.4,961.0,1029.,1099.,1172.,1247.,1327.,1414.,1509.,1610.,1716.,1825.,1936.,2050.,2164.,2280.,2397.,2514.,2631.,2749.,2867.,2985.,3104.,3222.,3341.,3460.])
    BaseLogPartFctRef[149]['NbLabels']=2
    BaseLogPartFctRef[149]['NbCliques']=2382
    BaseLogPartFctRef[149]['NbSites']=929
    BaseLogPartFctRef[149]['StdNgbhDivMoyNgbh']=0.1741


    #non-regular grids: Graphes15_02.pickle
    BaseLogPartFctRef[150]={}
    BaseLogPartFctRef[150]['LogPF']=np.array([25.65,26.93,28.25,29.60,30.99,32.41,33.86,35.35,36.87,38.44,40.04,41.67,43.34,45.05,46.79,48.59,50.45,52.32,54.24,56.22,58.24,60.28,62.37,64.50,66.68,68.87,71.09,73.35,75.62,77.93])
    BaseLogPartFctRef[150]['NbLabels']=2
    BaseLogPartFctRef[150]['NbCliques']=51
    BaseLogPartFctRef[150]['NbSites']=37
    BaseLogPartFctRef[150]['StdNgbhDivMoyNgbh']=0.4217
    BaseLogPartFctRef[151]={}
    BaseLogPartFctRef[151]['LogPF']=np.array([20.10,21.17,22.25,23.37,24.51,25.68,26.87,28.10,29.35,30.64,31.97,33.31,34.69,36.11,37.55,39.02,40.54,42.09,43.68,45.31,46.99,48.72,50.47,52.23,54.04,55.89,57.76,59.66,61.56,63.50])
    BaseLogPartFctRef[151]['NbLabels']=2
    BaseLogPartFctRef[151]['NbCliques']=42
    BaseLogPartFctRef[151]['NbSites']=29
    BaseLogPartFctRef[151]['StdNgbhDivMoyNgbh']=0.3948
    BaseLogPartFctRef[152]={}
    BaseLogPartFctRef[152]['LogPF']=np.array([24.26,25.45,26.66,27.91,29.19,30.50,31.84,33.21,34.61,36.04,37.50,39.00,40.55,42.12,43.73,45.37,47.04,48.76,50.51,52.30,54.13,55.99,57.89,59.84,61.81,63.80,65.83,67.90,69.98,72.09])
    BaseLogPartFctRef[152]['NbLabels']=2
    BaseLogPartFctRef[152]['NbCliques']=47
    BaseLogPartFctRef[152]['NbSites']=35
    BaseLogPartFctRef[152]['StdNgbhDivMoyNgbh']=0.4034
    BaseLogPartFctRef[153]={}
    BaseLogPartFctRef[153]['LogPF']=np.array([39.51,41.33,43.21,45.11,47.06,49.06,51.11,53.21,55.35,57.54,59.77,62.05,64.39,66.78,69.21,71.67,74.20,76.77,79.40,82.08,84.81,87.59,90.41,93.30,96.21,99.17,102.2,105.2,108.3,111.4])
    BaseLogPartFctRef[153]['NbLabels']=2
    BaseLogPartFctRef[153]['NbCliques']=72
    BaseLogPartFctRef[153]['NbSites']=57
    BaseLogPartFctRef[153]['StdNgbhDivMoyNgbh']=0.3793
    BaseLogPartFctRef[154]={}
    BaseLogPartFctRef[154]['LogPF']=np.array([30.50,31.89,33.32,34.77,36.27,37.81,39.37,40.97,42.61,44.28,45.99,47.74,49.53,51.36,53.23,55.14,57.08,59.06,61.09,63.16,65.27,67.41,69.59,71.81,74.05,76.33,78.65,80.99,83.36,85.77])
    BaseLogPartFctRef[154]['NbLabels']=2
    BaseLogPartFctRef[154]['NbCliques']=55
    BaseLogPartFctRef[154]['NbSites']=44
    BaseLogPartFctRef[154]['StdNgbhDivMoyNgbh']=0.3953
    BaseLogPartFctRef[155]={}
    BaseLogPartFctRef[155]['LogPF']=np.array([22.87,23.99,25.13,26.30,27.50,28.73,29.99,31.27,32.58,33.91,35.28,36.68,38.10,39.57,41.07,42.59,44.15,45.75,47.38,49.04,50.73,52.47,54.23,56.02,57.85,59.70,61.57,63.47,65.41,67.37])
    BaseLogPartFctRef[155]['NbLabels']=2
    BaseLogPartFctRef[155]['NbCliques']=44
    BaseLogPartFctRef[155]['NbSites']=33
    BaseLogPartFctRef[155]['StdNgbhDivMoyNgbh']=0.3655
    BaseLogPartFctRef[156]={}
    BaseLogPartFctRef[156]['LogPF']=np.array([34.66,36.61,38.60,40.65,42.75,44.88,47.07,49.32,51.63,53.99,56.41,58.90,61.43,64.04,66.73,69.50,72.34,75.27,78.27,81.36,84.51,87.73,91.00,94.34,97.73,101.2,104.6,108.2,111.7,115.3])
    BaseLogPartFctRef[156]['NbLabels']=2
    BaseLogPartFctRef[156]['NbCliques']=77
    BaseLogPartFctRef[156]['NbSites']=50
    BaseLogPartFctRef[156]['StdNgbhDivMoyNgbh']=0.3297
    BaseLogPartFctRef[157]={}
    BaseLogPartFctRef[157]['LogPF']=np.array([35.35,37.22,39.15,41.12,43.14,45.20,47.30,49.46,51.67,53.94,56.24,58.61,61.04,63.55,66.12,68.75,71.45,74.23,77.05,79.94,82.92,85.96,89.06,92.23,95.45,98.70,102.0,105.3,108.7,112.1])
    BaseLogPartFctRef[157]['NbLabels']=2
    BaseLogPartFctRef[157]['NbCliques']=74
    BaseLogPartFctRef[157]['NbSites']=51
    BaseLogPartFctRef[157]['StdNgbhDivMoyNgbh']=0.4138
    BaseLogPartFctRef[158]={}
    BaseLogPartFctRef[158]['LogPF']=np.array([15.94,16.75,17.59,18.44,19.30,20.19,21.10,22.03,22.99,23.96,24.97,25.99,27.04,28.12,29.21,30.34,31.51,32.69,33.90,35.13,36.39,37.68,39.00,40.33,41.69,43.08,44.48,45.91,47.36,48.81])
    BaseLogPartFctRef[158]['NbLabels']=2
    BaseLogPartFctRef[158]['NbCliques']=32
    BaseLogPartFctRef[158]['NbSites']=23
    BaseLogPartFctRef[158]['StdNgbhDivMoyNgbh']=0.3505
    BaseLogPartFctRef[159]={}
    BaseLogPartFctRef[159]['LogPF']=np.array([38.82,40.84,42.91,45.04,47.22,49.44,51.71,54.04,56.44,58.89,61.38,63.94,66.57,69.27,72.03,74.84,77.74,80.71,83.73,86.83,90.00,93.25,96.57,99.94,103.3,106.8,110.3,113.9,117.5,121.1])
    BaseLogPartFctRef[159]['NbLabels']=2
    BaseLogPartFctRef[159]['NbCliques']=80
    BaseLogPartFctRef[159]['NbSites']=56
    BaseLogPartFctRef[159]['StdNgbhDivMoyNgbh']=0.4079
    BaseLogPartFctRef[160]={}
    BaseLogPartFctRef[160]['LogPF']=np.array([22.18,23.27,24.39,25.54,26.71,27.91,29.13,30.39,31.67,32.98,34.33,35.71,37.11,38.55,40.02,41.54,43.09,44.68,46.31,47.97,49.67,51.39,53.15,54.94,56.76,58.61,60.48,62.38,64.30,66.24])
    BaseLogPartFctRef[160]['NbLabels']=2
    BaseLogPartFctRef[160]['NbCliques']=43
    BaseLogPartFctRef[160]['NbSites']=32
    BaseLogPartFctRef[160]['StdNgbhDivMoyNgbh']=0.4074
    BaseLogPartFctRef[161]={}
    BaseLogPartFctRef[161]['LogPF']=np.array([31.19,32.66,34.17,35.71,37.28,38.89,40.54,42.23,43.95,45.71,47.51,49.36,51.24,53.17,55.13,57.14,59.19,61.26,63.38,65.55,67.76,70.01,72.30,74.63,76.99,79.38,81.83,84.29,86.80,89.34])
    BaseLogPartFctRef[161]['NbLabels']=2
    BaseLogPartFctRef[161]['NbCliques']=58
    BaseLogPartFctRef[161]['NbSites']=45
    BaseLogPartFctRef[161]['StdNgbhDivMoyNgbh']=0.3792
    BaseLogPartFctRef[162]={}
    BaseLogPartFctRef[162]['LogPF']=np.array([21.49,22.58,23.69,24.83,26.00,27.20,28.41,29.66,30.94,32.26,33.60,34.97,36.38,37.82,39.30,40.81,42.35,43.94,45.57,47.24,48.93,50.65,52.43,54.24,56.06,57.92,59.81,61.71,63.63,65.57])
    BaseLogPartFctRef[162]['NbLabels']=2
    BaseLogPartFctRef[162]['NbCliques']=43
    BaseLogPartFctRef[162]['NbSites']=31
    BaseLogPartFctRef[162]['StdNgbhDivMoyNgbh']=0.4579
    BaseLogPartFctRef[163]={}
    BaseLogPartFctRef[163]['LogPF']=np.array([33.27,34.93,36.65,38.40,40.19,42.03,43.90,45.83,47.80,49.81,51.87,53.99,56.15,58.37,60.64,62.97,65.34,67.79,70.27,72.82,75.44,78.13,80.86,83.65,86.46,89.32,92.22,95.17,98.12,101.1])
    BaseLogPartFctRef[163]['NbLabels']=2
    BaseLogPartFctRef[163]['NbCliques']=66
    BaseLogPartFctRef[163]['NbSites']=48
    BaseLogPartFctRef[163]['StdNgbhDivMoyNgbh']=0.4269
    BaseLogPartFctRef[164]={}
    BaseLogPartFctRef[164]['LogPF']=np.array([26.34,27.64,28.96,30.31,31.71,33.13,34.58,36.07,37.60,39.16,40.76,42.39,44.06,45.76,47.51,49.29,51.12,52.97,54.86,56.78,58.75,60.76,62.81,64.89,67.01,69.16,71.34,73.54,75.78,78.04])
    BaseLogPartFctRef[164]['NbLabels']=2
    BaseLogPartFctRef[164]['NbCliques']=51
    BaseLogPartFctRef[164]['NbSites']=38
    BaseLogPartFctRef[164]['StdNgbhDivMoyNgbh']=0.4086
    BaseLogPartFctRef[165]={}
    BaseLogPartFctRef[165]['LogPF']=np.array([31.88,33.32,34.80,36.31,37.87,39.46,41.08,42.75,44.44,46.17,47.94,49.75,51.60,53.49,55.42,57.40,59.42,61.47,63.57,65.71,67.88,70.10,72.35,74.65,76.96,79.32,81.70,84.13,86.58,89.07])
    BaseLogPartFctRef[165]['NbLabels']=2
    BaseLogPartFctRef[165]['NbCliques']=57
    BaseLogPartFctRef[165]['NbSites']=46
    BaseLogPartFctRef[165]['StdNgbhDivMoyNgbh']=0.3925
    BaseLogPartFctRef[166]={}
    BaseLogPartFctRef[166]['LogPF']=np.array([29.81,31.35,32.93,34.55,36.21,37.90,39.64,41.42,43.24,45.10,47.01,48.97,50.96,53.01,55.10,57.25,59.43,61.66,63.96,66.30,68.69,71.13,73.62,76.16,78.73,81.34,84.01,86.69,89.40,92.15])
    BaseLogPartFctRef[166]['NbLabels']=2
    BaseLogPartFctRef[166]['NbCliques']=61
    BaseLogPartFctRef[166]['NbSites']=43
    BaseLogPartFctRef[166]['StdNgbhDivMoyNgbh']=0.3632
    BaseLogPartFctRef[167]={}
    BaseLogPartFctRef[167]['LogPF']=np.array([27.03,28.45,29.90,31.39,32.91,34.47,36.07,37.71,39.38,41.10,42.86,44.67,46.52,48.41,50.35,52.33,54.36,56.45,58.57,60.76,63.02,65.31,67.66,70.05,72.47,74.93,77.42,79.93,82.48,85.05])
    BaseLogPartFctRef[167]['NbLabels']=2
    BaseLogPartFctRef[167]['NbCliques']=56
    BaseLogPartFctRef[167]['NbSites']=39
    BaseLogPartFctRef[167]['StdNgbhDivMoyNgbh']=0.4150
    BaseLogPartFctRef[168]={}
    BaseLogPartFctRef[168]['LogPF']=np.array([41.59,43.57,45.59,47.67,49.80,51.96,54.20,56.47,58.80,61.17,63.61,66.09,68.63,71.23,73.89,76.61,79.37,82.21,85.12,88.09,91.13,94.22,97.35,100.6,103.8,107.1,110.4,113.8,117.2,120.7])
    BaseLogPartFctRef[168]['NbLabels']=2
    BaseLogPartFctRef[168]['NbCliques']=78
    BaseLogPartFctRef[168]['NbSites']=60
    BaseLogPartFctRef[168]['StdNgbhDivMoyNgbh']=0.4212
    BaseLogPartFctRef[169]={}
    BaseLogPartFctRef[169]['LogPF']=np.array([36.04,37.77,39.53,41.34,43.18,45.06,47.00,48.97,51.00,53.07,55.18,57.35,59.55,61.82,64.14,66.52,68.95,71.42,73.96,76.56,79.20,81.90,84.65,87.44,90.26,93.14,96.06,99.00,102.0,105.0])
    BaseLogPartFctRef[169]['NbLabels']=2
    BaseLogPartFctRef[169]['NbCliques']=68
    BaseLogPartFctRef[169]['NbSites']=52
    BaseLogPartFctRef[169]['StdNgbhDivMoyNgbh']=0.3902
    BaseLogPartFctRef[170]={}
    BaseLogPartFctRef[170]['LogPF']=np.array([38.12,40.25,42.43,44.67,46.95,49.29,51.70,54.15,56.67,59.24,61.88,64.59,67.36,70.21,73.14,76.15,79.24,82.40,85.67,89.02,92.45,95.95,99.51,103.1,106.8,110.6,114.4,118.2,122.1,126.1])
    BaseLogPartFctRef[170]['NbLabels']=2
    BaseLogPartFctRef[170]['NbCliques']=84
    BaseLogPartFctRef[170]['NbSites']=55
    BaseLogPartFctRef[170]['StdNgbhDivMoyNgbh']=0.3897
    BaseLogPartFctRef[171]={}
    BaseLogPartFctRef[171]['LogPF']=np.array([33.96,35.58,37.24,38.94,40.68,42.46,44.29,46.15,48.06,50.01,52.00,54.05,56.14,58.26,60.45,62.68,64.97,67.30,69.69,72.12,74.60,77.14,79.72,82.32,84.98,87.66,90.40,93.16,95.96,98.79])
    BaseLogPartFctRef[171]['NbLabels']=2
    BaseLogPartFctRef[171]['NbCliques']=64
    BaseLogPartFctRef[171]['NbSites']=49
    BaseLogPartFctRef[171]['StdNgbhDivMoyNgbh']=0.4340
    BaseLogPartFctRef[172]={}
    BaseLogPartFctRef[172]['LogPF']=np.array([24.26,25.32,26.41,27.53,28.67,29.84,31.03,32.26,33.52,34.80,36.11,37.43,38.80,40.19,41.61,43.06,44.54,46.04,47.59,49.15,50.75,52.37,54.02,55.69,57.39,59.12,60.86,62.62,64.42,66.23])
    BaseLogPartFctRef[172]['NbLabels']=2
    BaseLogPartFctRef[172]['NbCliques']=42
    BaseLogPartFctRef[172]['NbSites']=35
    BaseLogPartFctRef[172]['StdNgbhDivMoyNgbh']=0.4128
    BaseLogPartFctRef[173]={}
    BaseLogPartFctRef[173]['LogPF']=np.array([42.98,45.15,47.38,49.66,51.99,54.38,56.82,59.33,61.88,64.50,67.18,69.93,72.73,75.62,78.57,81.58,84.68,87.85,91.09,94.42,97.82,101.3,104.8,108.4,112.1,115.8,119.5,123.3,127.2,131.1])
    BaseLogPartFctRef[173]['NbLabels']=2
    BaseLogPartFctRef[173]['NbCliques']=86
    BaseLogPartFctRef[173]['NbSites']=62
    BaseLogPartFctRef[173]['StdNgbhDivMoyNgbh']=0.3737
    BaseLogPartFctRef[174]={}
    BaseLogPartFctRef[174]['LogPF']=np.array([22.87,23.98,25.12,26.29,27.48,28.70,29.96,31.23,32.55,33.89,35.27,36.67,38.10,39.58,41.07,42.60,44.17,45.78,47.43,49.10,50.81,52.56,54.33,56.13,57.97,59.84,61.73,63.66,65.61,67.58])
    BaseLogPartFctRef[174]['NbLabels']=2
    BaseLogPartFctRef[174]['NbCliques']=44
    BaseLogPartFctRef[174]['NbSites']=33
    BaseLogPartFctRef[174]['StdNgbhDivMoyNgbh']=0.4262
    BaseLogPartFctRef[175]={}
    BaseLogPartFctRef[175]['LogPF']=np.array([58.92,61.86,64.87,67.95,71.11,74.34,77.66,81.05,84.50,88.03,91.63,95.36,99.13,103.0,107.0,111.0,115.2,119.4,123.8,128.2,132.7,137.3,142.0,146.8,151.6,156.6,161.6,166.6,171.7,176.9])
    BaseLogPartFctRef[175]['NbLabels']=2
    BaseLogPartFctRef[175]['NbCliques']=116
    BaseLogPartFctRef[175]['NbSites']=85
    BaseLogPartFctRef[175]['StdNgbhDivMoyNgbh']=0.4058
    BaseLogPartFctRef[176]={}
    BaseLogPartFctRef[176]['LogPF']=np.array([26.34,27.94,29.57,31.24,32.96,34.71,36.51,38.36,40.24,42.18,44.16,46.22,48.32,50.50,52.74,55.05,57.42,59.86,62.39,64.97,67.61,70.32,73.08,75.89,78.74,81.63,84.55,87.50,90.47,93.46])
    BaseLogPartFctRef[176]['NbLabels']=2
    BaseLogPartFctRef[176]['NbCliques']=63
    BaseLogPartFctRef[176]['NbSites']=38
    BaseLogPartFctRef[176]['StdNgbhDivMoyNgbh']=0.3238
    BaseLogPartFctRef[177]={}
    BaseLogPartFctRef[177]['LogPF']=np.array([33.27,35.14,37.06,39.03,41.04,43.10,45.21,47.38,49.60,51.88,54.21,56.62,59.05,61.58,64.19,66.88,69.65,72.49,75.44,78.46,81.53,84.66,87.85,91.10,94.39,97.73,101.1,104.5,107.9,111.4])
    BaseLogPartFctRef[177]['NbLabels']=2
    BaseLogPartFctRef[177]['NbCliques']=74
    BaseLogPartFctRef[177]['NbSites']=48
    BaseLogPartFctRef[177]['StdNgbhDivMoyNgbh']=0.4340
    BaseLogPartFctRef[178]={}
    BaseLogPartFctRef[178]['LogPF']=np.array([24.95,26.22,27.52,28.84,30.20,31.60,33.02,34.48,35.98,37.51,39.07,40.67,42.32,44.00,45.72,47.48,49.28,51.14,53.03,54.96,56.94,58.94,61.00,63.08,65.20,67.35,69.54,71.75,73.99,76.25])
    BaseLogPartFctRef[178]['NbLabels']=2
    BaseLogPartFctRef[178]['NbCliques']=50
    BaseLogPartFctRef[178]['NbSites']=36
    BaseLogPartFctRef[178]['StdNgbhDivMoyNgbh']=0.3966
    BaseLogPartFctRef[179]={}
    BaseLogPartFctRef[179]['LogPF']=np.array([27.03,28.32,29.64,31.00,32.38,33.81,35.26,36.74,38.26,39.82,41.41,43.02,44.68,46.38,48.11,49.88,51.69,53.55,55.43,57.36,59.31,61.31,63.36,65.44,67.52,69.67,71.83,74.03,76.25,78.49])
    BaseLogPartFctRef[179]['NbLabels']=2
    BaseLogPartFctRef[179]['NbCliques']=51
    BaseLogPartFctRef[179]['NbSites']=39
    BaseLogPartFctRef[179]['StdNgbhDivMoyNgbh']=0.3922


    #non-regular grids: Graphes15_03.pickle
    BaseLogPartFctRef[180]={}
    BaseLogPartFctRef[180]['LogPF']=np.array([701.5,748.9,797.5,847.2,898.2,950.3,1004.,1059.,1115.,1172.,1232.,1293.,1355.,1420.,1488.,1558.,1632.,1708.,1787.,1868.,1951.,2036.,2122.,2209.,2297.,2386.,2475.,2565.,2655.,2746.])
    BaseLogPartFctRef[180]['NbLabels']=2
    BaseLogPartFctRef[180]['NbCliques']=1872
    BaseLogPartFctRef[180]['NbSites']=1012
    BaseLogPartFctRef[180]['StdNgbhDivMoyNgbh']=0.3409
    BaseLogPartFctRef[181]={}
    BaseLogPartFctRef[181]['LogPF']=np.array([385.4,410.2,435.7,461.8,488.4,515.8,543.8,572.5,601.9,632.1,663.0,694.9,727.6,761.4,796.2,832.3,869.8,908.5,948.6,989.9,1032.,1076.,1120.,1165.,1210.,1256.,1303.,1349.,1396.,1444.])
    BaseLogPartFctRef[181]['NbLabels']=2
    BaseLogPartFctRef[181]['NbCliques']=981
    BaseLogPartFctRef[181]['NbSites']=556
    BaseLogPartFctRef[181]['StdNgbhDivMoyNgbh']=0.3564
    BaseLogPartFctRef[182]={}
    BaseLogPartFctRef[182]['LogPF']=np.array([524.7,560.5,597.2,634.8,673.2,712.6,752.9,794.3,836.7,880.2,925.0,971.0,1019.,1068.,1119.,1173.,1229.,1287.,1348.,1410.,1473.,1537.,1602.,1668.,1734.,1801.,1869.,1937.,2005.,2074.])
    BaseLogPartFctRef[182]['NbLabels']=2
    BaseLogPartFctRef[182]['NbCliques']=1414
    BaseLogPartFctRef[182]['NbSites']=757
    BaseLogPartFctRef[182]['StdNgbhDivMoyNgbh']=0.3404
    BaseLogPartFctRef[183]={}
    BaseLogPartFctRef[183]['LogPF']=np.array([406.9,432.3,458.4,485.1,512.5,540.6,569.3,598.7,628.8,659.7,691.4,723.9,757.5,792.2,827.9,865.0,903.4,943.1,984.1,1026.,1069.,1113.,1158.,1204.,1250.,1297.,1344.,1391.,1439.,1487.])
    BaseLogPartFctRef[183]['NbLabels']=2
    BaseLogPartFctRef[183]['NbCliques']=1006
    BaseLogPartFctRef[183]['NbSites']=587
    BaseLogPartFctRef[183]['StdNgbhDivMoyNgbh']=0.3831
    BaseLogPartFctRef[184]={}
    BaseLogPartFctRef[184]['LogPF']=np.array([262.0,279.4,297.2,315.5,334.1,353.2,372.7,392.8,413.4,434.5,456.1,478.4,501.4,525.3,550.1,575.8,602.4,630.1,658.6,688.0,718.1,748.7,779.9,811.5,843.5,875.7,908.2,940.9,973.8,1007.])
    BaseLogPartFctRef[184]['NbLabels']=2
    BaseLogPartFctRef[184]['NbCliques']=686
    BaseLogPartFctRef[184]['NbSites']=378
    BaseLogPartFctRef[184]['StdNgbhDivMoyNgbh']=0.3491
    BaseLogPartFctRef[185]={}
    BaseLogPartFctRef[185]['LogPF']=np.array([492.1,524.5,557.6,591.5,626.2,661.7,698.2,735.5,773.7,813.0,853.2,894.6,937.3,981.3,1027.,1074.,1123.,1174.,1227.,1281.,1337.,1394.,1452.,1511.,1570.,1630.,1691.,1751.,1813.,1874.])
    BaseLogPartFctRef[185]['NbLabels']=2
    BaseLogPartFctRef[185]['NbCliques']=1276
    BaseLogPartFctRef[185]['NbSites']=710
    BaseLogPartFctRef[185]['StdNgbhDivMoyNgbh']=0.3444
    BaseLogPartFctRef[186]={}
    BaseLogPartFctRef[186]['LogPF']=np.array([256.5,273.2,290.3,307.8,325.8,344.2,363.0,382.3,402.1,422.4,443.3,464.7,486.7,509.5,533.1,557.7,583.2,609.7,637.1,665.4,694.4,723.9,754.0,784.4,815.1,846.2,877.5,908.9,940.5,972.3])
    BaseLogPartFctRef[186]['NbLabels']=2
    BaseLogPartFctRef[186]['NbCliques']=660
    BaseLogPartFctRef[186]['NbSites']=370
    BaseLogPartFctRef[186]['StdNgbhDivMoyNgbh']=0.3615
    BaseLogPartFctRef[187]={}
    BaseLogPartFctRef[187]['LogPF']=np.array([407.6,434.7,462.6,491.1,520.3,550.3,581.0,612.4,644.7,677.7,711.7,746.7,782.8,820.2,858.9,899.2,941.1,984.8,1030.,1076.,1123.,1172.,1221.,1270.,1321.,1371.,1422.,1473.,1525.,1577.])
    BaseLogPartFctRef[187]['NbLabels']=2
    BaseLogPartFctRef[187]['NbCliques']=1074
    BaseLogPartFctRef[187]['NbSites']=588
    BaseLogPartFctRef[187]['StdNgbhDivMoyNgbh']=0.3498
    BaseLogPartFctRef[188]={}
    BaseLogPartFctRef[188]['LogPF']=np.array([283.5,299.9,316.7,334.0,351.7,369.7,388.3,407.2,426.6,446.5,466.9,487.9,509.3,531.4,554.0,577.3,601.3,625.9,651.3,677.4,704.1,731.3,759.1,787.4,816.1,845.2,874.7,904.5,934.6,965.0])
    BaseLogPartFctRef[188]['NbLabels']=2
    BaseLogPartFctRef[188]['NbCliques']=649
    BaseLogPartFctRef[188]['NbSites']=409
    BaseLogPartFctRef[188]['StdNgbhDivMoyNgbh']=0.3758
    BaseLogPartFctRef[189]={}
    BaseLogPartFctRef[189]['LogPF']=np.array([263.4,281.2,299.4,318.1,337.2,356.8,376.8,397.4,418.5,440.2,462.4,485.3,509.1,533.7,559.2,585.8,613.6,642.5,672.3,702.9,734.2,766.0,798.3,831.0,864.0,897.3,930.8,964.5,998.4,1032.])
    BaseLogPartFctRef[189]['NbLabels']=2
    BaseLogPartFctRef[189]['NbCliques']=703
    BaseLogPartFctRef[189]['NbSites']=380
    BaseLogPartFctRef[189]['StdNgbhDivMoyNgbh']=0.3573
    BaseLogPartFctRef[190]={}
    BaseLogPartFctRef[190]['LogPF']=np.array([467.9,499.4,531.7,564.7,598.6,633.3,668.8,705.2,742.6,780.9,820.3,860.8,902.7,946.0,991.0,1038.,1087.,1138.,1191.,1245.,1300.,1356.,1413.,1471.,1529.,1588.,1647.,1707.,1767.,1827.])
    BaseLogPartFctRef[190]['NbLabels']=2
    BaseLogPartFctRef[190]['NbCliques']=1244
    BaseLogPartFctRef[190]['NbSites']=675
    BaseLogPartFctRef[190]['StdNgbhDivMoyNgbh']=0.3489
    BaseLogPartFctRef[191]={}
    BaseLogPartFctRef[191]['LogPF']=np.array([519.2,553.9,589.5,625.8,663.2,701.5,740.7,780.8,821.9,864.1,907.5,952.2,998.2,1046.,1095.,1147.,1201.,1257.,1315.,1374.,1435.,1497.,1559.,1623.,1687.,1752.,1817.,1883.,1949.,2015.])
    BaseLogPartFctRef[191]['NbLabels']=2
    BaseLogPartFctRef[191]['NbCliques']=1371
    BaseLogPartFctRef[191]['NbSites']=749
    BaseLogPartFctRef[191]['StdNgbhDivMoyNgbh']=0.3458
    BaseLogPartFctRef[192]={}
    BaseLogPartFctRef[192]['LogPF']=np.array([434.6,462.4,490.8,520.0,549.9,580.4,611.7,643.8,676.6,710.4,745.0,780.6,817.3,855.1,894.2,934.8,976.9,1020.,1065.,1112.,1159.,1207.,1256.,1306.,1357.,1408.,1459.,1511.,1564.,1616.])
    BaseLogPartFctRef[192]['NbLabels']=2
    BaseLogPartFctRef[192]['NbCliques']=1097
    BaseLogPartFctRef[192]['NbSites']=627
    BaseLogPartFctRef[192]['StdNgbhDivMoyNgbh']=0.3576
    BaseLogPartFctRef[193]={}
    BaseLogPartFctRef[193]['LogPF']=np.array([264.8,283.0,301.6,320.6,340.2,360.2,380.7,401.7,423.3,445.5,468.2,491.7,516.0,541.4,567.8,595.3,624.2,653.9,684.5,715.9,748.0,780.6,813.7,847.1,880.9,914.9,949.2,983.7,1018.,1053.])
    BaseLogPartFctRef[193]['NbLabels']=2
    BaseLogPartFctRef[193]['NbCliques']=717
    BaseLogPartFctRef[193]['NbSites']=382
    BaseLogPartFctRef[193]['StdNgbhDivMoyNgbh']=0.3624
    BaseLogPartFctRef[194]={}
    BaseLogPartFctRef[194]['LogPF']=np.array([480.4,513.2,546.8,581.3,616.6,652.7,689.7,727.7,766.7,806.7,848.0,890.5,934.5,980.4,1028.,1078.,1131.,1185.,1241.,1298.,1356.,1415.,1475.,1536.,1597.,1659.,1720.,1783.,1845.,1908.])
    BaseLogPartFctRef[194]['NbLabels']=2
    BaseLogPartFctRef[194]['NbCliques']=1297
    BaseLogPartFctRef[194]['NbSites']=693
    BaseLogPartFctRef[194]['StdNgbhDivMoyNgbh']=0.3674
    BaseLogPartFctRef[195]={}
    BaseLogPartFctRef[195]['LogPF']=np.array([228.7,242.7,257.0,271.7,286.7,302.1,317.9,334.0,350.5,367.4,384.8,402.7,420.9,439.7,459.2,479.2,499.9,521.3,543.3,566.1,589.4,613.2,637.5,662.3,687.4,712.9,738.5,764.4,790.5,816.7])
    BaseLogPartFctRef[195]['NbLabels']=2
    BaseLogPartFctRef[195]['NbCliques']=552
    BaseLogPartFctRef[195]['NbSites']=330
    BaseLogPartFctRef[195]['StdNgbhDivMoyNgbh']=0.3641
    BaseLogPartFctRef[196]={}
    BaseLogPartFctRef[196]['LogPF']=np.array([406.9,433.7,461.2,489.4,518.2,547.8,578.0,609.0,640.7,673.4,706.9,741.3,776.9,813.5,851.5,891.0,932.2,974.7,1019.,1064.,1111.,1158.,1207.,1255.,1305.,1355.,1405.,1456.,1507.,1558.])
    BaseLogPartFctRef[196]['NbLabels']=2
    BaseLogPartFctRef[196]['NbCliques']=1060
    BaseLogPartFctRef[196]['NbSites']=587
    BaseLogPartFctRef[196]['StdNgbhDivMoyNgbh']=0.3567
    BaseLogPartFctRef[197]={}
    BaseLogPartFctRef[197]['LogPF']=np.array([381.2,406.3,431.9,458.2,485.1,512.6,540.8,569.7,599.3,629.7,660.9,693.0,726.1,760.3,795.6,832.3,870.4,910.0,951.0,993.2,1036.,1080.,1125.,1171.,1217.,1263.,1310.,1357.,1404.,1452.])
    BaseLogPartFctRef[197]['NbLabels']=2
    BaseLogPartFctRef[197]['NbCliques']=988
    BaseLogPartFctRef[197]['NbSites']=550
    BaseLogPartFctRef[197]['StdNgbhDivMoyNgbh']=0.3431
    BaseLogPartFctRef[198]={}
    BaseLogPartFctRef[198]['LogPF']=np.array([544.1,580.1,617.0,654.8,693.6,733.2,773.9,815.4,858.0,901.7,946.5,992.7,1040.,1089.,1140.,1193.,1248.,1305.,1364.,1425.,1488.,1552.,1617.,1683.,1749.,1816.,1884.,1952.,2020.,2089.])
    BaseLogPartFctRef[198]['NbLabels']=2
    BaseLogPartFctRef[198]['NbCliques']=1423
    BaseLogPartFctRef[198]['NbSites']=785
    BaseLogPartFctRef[198]['StdNgbhDivMoyNgbh']=0.3462
    BaseLogPartFctRef[199]={}
    BaseLogPartFctRef[199]['LogPF']=np.array([262.7,279.9,297.6,315.7,334.2,353.2,372.7,392.6,413.0,433.9,455.5,477.7,500.4,524.1,548.5,574.1,600.5,627.9,656.1,685.2,714.9,745.3,776.2,807.4,839.0,871.0,903.2,935.5,968.1,1001.])
    BaseLogPartFctRef[199]['NbLabels']=2
    BaseLogPartFctRef[199]['NbCliques']=681
    BaseLogPartFctRef[199]['NbSites']=379
    BaseLogPartFctRef[199]['StdNgbhDivMoyNgbh']=0.3542
    BaseLogPartFctRef[200]={}
    BaseLogPartFctRef[200]['LogPF']=np.array([150.4,159.8,169.4,179.3,189.4,199.7,210.3,221.1,232.2,243.6,255.3,267.3,279.6,292.4,305.6,319.1,333.2,347.7,362.6,378.1,393.8,410.0,426.6,443.4,460.4,477.6,494.9,512.5,530.1,547.8])
    BaseLogPartFctRef[200]['NbLabels']=2
    BaseLogPartFctRef[200]['NbCliques']=371
    BaseLogPartFctRef[200]['NbSites']=217
    BaseLogPartFctRef[200]['StdNgbhDivMoyNgbh']=0.3551
    BaseLogPartFctRef[201]={}
    BaseLogPartFctRef[201]['LogPF']=np.array([286.3,305.2,324.6,344.5,364.8,385.7,407.0,428.9,451.4,474.4,498.1,522.4,547.6,573.5,600.8,629.2,658.7,689.5,720.9,753.3,786.3,819.8,853.9,888.4,923.2,958.3,993.7,1029.,1065.,1101.])
    BaseLogPartFctRef[201]['NbLabels']=2
    BaseLogPartFctRef[201]['NbCliques']=748
    BaseLogPartFctRef[201]['NbSites']=413
    BaseLogPartFctRef[201]['StdNgbhDivMoyNgbh']=0.3603
    BaseLogPartFctRef[202]={}
    BaseLogPartFctRef[202]['LogPF']=np.array([490.1,522.9,556.7,591.2,626.5,662.8,699.9,738.0,776.9,817.0,858.1,900.5,944.0,989.1,1036.,1084.,1135.,1187.,1242.,1298.,1355.,1413.,1473.,1533.,1594.,1656.,1718.,1780.,1843.,1906.])
    BaseLogPartFctRef[202]['NbLabels']=2
    BaseLogPartFctRef[202]['NbCliques']=1300
    BaseLogPartFctRef[202]['NbSites']=707
    BaseLogPartFctRef[202]['StdNgbhDivMoyNgbh']=0.3424
    BaseLogPartFctRef[203]={}
    BaseLogPartFctRef[203]['LogPF']=np.array([433.9,461.4,489.6,518.5,548.1,578.3,609.4,641.2,673.7,707.1,741.4,776.6,812.9,850.3,889.1,929.2,970.8,1014.,1058.,1104.,1151.,1198.,1247.,1296.,1346.,1397.,1448.,1499.,1551.,1603.])
    BaseLogPartFctRef[203]['NbLabels']=2
    BaseLogPartFctRef[203]['NbCliques']=1087
    BaseLogPartFctRef[203]['NbSites']=626
    BaseLogPartFctRef[203]['StdNgbhDivMoyNgbh']=0.3691
    BaseLogPartFctRef[204]={}
    BaseLogPartFctRef[204]['LogPF']=np.array([287.7,306.4,325.5,345.1,365.2,385.8,406.8,428.4,450.5,473.2,496.6,520.7,545.5,571.2,598.0,625.7,654.5,684.3,715.0,746.4,778.5,811.2,844.6,878.3,912.4,946.9,981.8,1017.,1052.,1088.])
    BaseLogPartFctRef[204]['NbLabels']=2
    BaseLogPartFctRef[204]['NbCliques']=738
    BaseLogPartFctRef[204]['NbSites']=415
    BaseLogPartFctRef[204]['StdNgbhDivMoyNgbh']=0.3716
    BaseLogPartFctRef[205]={}
    BaseLogPartFctRef[205]['LogPF']=np.array([294.6,314.1,334.1,354.5,375.6,397.0,419.0,441.5,464.5,488.3,512.7,537.8,563.8,590.7,618.6,647.9,678.3,709.6,741.9,775.1,808.9,843.4,878.3,913.8,949.6,985.7,1022.,1059.,1096.,1133.])
    BaseLogPartFctRef[205]['NbLabels']=2
    BaseLogPartFctRef[205]['NbCliques']=770
    BaseLogPartFctRef[205]['NbSites']=425
    BaseLogPartFctRef[205]['StdNgbhDivMoyNgbh']=0.3628
    BaseLogPartFctRef[206]={}
    BaseLogPartFctRef[206]['LogPF']=np.array([334.1,356.8,380.0,403.8,428.1,453.0,478.6,504.8,531.7,559.3,587.6,616.7,646.9,678.1,710.3,743.9,779.0,815.4,853.2,892.2,932.0,972.6,1014.,1056.,1098.,1140.,1183.,1226.,1270.,1313.])
    BaseLogPartFctRef[206]['NbLabels']=2
    BaseLogPartFctRef[206]['NbCliques']=895
    BaseLogPartFctRef[206]['NbSites']=482
    BaseLogPartFctRef[206]['StdNgbhDivMoyNgbh']=0.3485
    BaseLogPartFctRef[207]={}
    BaseLogPartFctRef[207]['LogPF']=np.array([627.3,669.2,712.2,756.2,801.3,847.4,894.7,943.2,992.8,1044.,1096.,1150.,1205.,1263.,1322.,1384.,1448.,1515.,1584.,1655.,1728.,1802.,1878.,1955.,2032.,2110.,2189.,2268.,2348.,2428.])
    BaseLogPartFctRef[207]['NbLabels']=2
    BaseLogPartFctRef[207]['NbCliques']=1656
    BaseLogPartFctRef[207]['NbSites']=905
    BaseLogPartFctRef[207]['StdNgbhDivMoyNgbh']=0.3428
    BaseLogPartFctRef[208]={}
    BaseLogPartFctRef[208]['LogPF']=np.array([339.6,362.1,385.1,408.7,432.8,457.5,482.9,508.8,535.4,562.7,590.7,619.4,649.0,679.5,711.3,744.3,778.6,814.3,851.3,889.5,928.4,968.2,1009.,1049.,1091.,1133.,1175.,1217.,1260.,1302.])
    BaseLogPartFctRef[208]['NbLabels']=2
    BaseLogPartFctRef[208]['NbCliques']=887
    BaseLogPartFctRef[208]['NbSites']=490
    BaseLogPartFctRef[208]['StdNgbhDivMoyNgbh']=0.3409
    BaseLogPartFctRef[209]={}
    BaseLogPartFctRef[209]['LogPF']=np.array([333.4,356.1,379.3,403.0,427.4,452.3,477.9,504.1,530.9,558.5,586.8,616.0,646.2,677.5,710.2,744.3,779.8,816.7,854.8,894.1,934.2,974.8,1016.,1058.,1100.,1142.,1185.,1228.,1271.,1314.])
    BaseLogPartFctRef[209]['NbLabels']=2
    BaseLogPartFctRef[209]['NbCliques']=895
    BaseLogPartFctRef[209]['NbSites']=481
    BaseLogPartFctRef[209]['StdNgbhDivMoyNgbh']=0.3583

    #non-regular grids: Graphes15_05.pickle
    BaseLogPartFctRef[210]={}
    BaseLogPartFctRef[210]['LogPF']=np.array([2086.,2284.,2487.,2694.,2907.,3125.,3349.,3579.,3817.,4070.,4350.,4649.,4971.,5308.,5659.,6020.,6388.,6761.,7138.,7518.,7901.,8285.,8670.,9057.,9444.,9832.,10220.,10609.,10998.,11387.])
    BaseLogPartFctRef[210]['NbLabels']=2
    BaseLogPartFctRef[210]['NbCliques']=7810
    BaseLogPartFctRef[210]['NbSites']=3010
    BaseLogPartFctRef[210]['StdNgbhDivMoyNgbh']=0.1767
    BaseLogPartFctRef[211]={}
    BaseLogPartFctRef[211]['LogPF']=np.array([1987.,2171.,2360.,2554.,2753.,2956.,3165.,3380.,3602.,3836.,4095.,4368.,4663.,4974.,5298.,5631.,5972.,6318.,6668.,7021.,7377.,7735.,8094.,8454.,8814.,9176.,9538.,9900.,10263.,10626.])
    BaseLogPartFctRef[211]['NbLabels']=2
    BaseLogPartFctRef[211]['NbCliques']=7288
    BaseLogPartFctRef[211]['NbSites']=2866
    BaseLogPartFctRef[211]['StdNgbhDivMoyNgbh']=0.1894
    BaseLogPartFctRef[212]={}
    BaseLogPartFctRef[212]['LogPF']=np.array([2013.,2199.,2389.,2584.,2784.,2988.,3199.,3415.,3638.,3871.,4129.,4407.,4702.,5012.,5337.,5672.,6015.,6363.,6715.,7070.,7428.,7787.,8148.,8511.,8874.,9237.,9602.,9966.,10332.,10697.])
    BaseLogPartFctRef[212]['NbLabels']=2
    BaseLogPartFctRef[212]['NbCliques']=7335
    BaseLogPartFctRef[212]['NbSites']=2904
    BaseLogPartFctRef[212]['StdNgbhDivMoyNgbh']=0.1867
    BaseLogPartFctRef[213]={}
    BaseLogPartFctRef[213]['LogPF']=np.array([2043.,2233.,2427.,2627.,2832.,3041.,3257.,3478.,3707.,3952.,4217.,4501.,4807.,5127.,5461.,5805.,6156.,6513.,6874.,7239.,7605.,7974.,8344.,8715.,9086.,9459.,9832.,10205.,10579.,10953.])
    BaseLogPartFctRef[213]['NbLabels']=2
    BaseLogPartFctRef[213]['NbCliques']=7509
    BaseLogPartFctRef[213]['NbSites']=2947
    BaseLogPartFctRef[213]['StdNgbhDivMoyNgbh']=0.1871
    BaseLogPartFctRef[214]={}
    BaseLogPartFctRef[214]['LogPF']=np.array([2050.,2242.,2438.,2639.,2845.,3056.,3273.,3496.,3726.,3970.,4236.,4527.,4836.,5161.,5498.,5846.,6201.,6561.,6926.,7293.,7663.,8034.,8407.,8781.,9156.,9531.,9907.,10284.,10660.,11037.])
    BaseLogPartFctRef[214]['NbLabels']=2
    BaseLogPartFctRef[214]['NbCliques']=7564
    BaseLogPartFctRef[214]['NbSites']=2958
    BaseLogPartFctRef[214]['StdNgbhDivMoyNgbh']=0.1783
    BaseLogPartFctRef[215]={}
    BaseLogPartFctRef[215]['LogPF']=np.array([2028.,2218.,2412.,2612.,2816.,3025.,3240.,3461.,3690.,3934.,4197.,4483.,4788.,5109.,5443.,5787.,6138.,6495.,6857.,7221.,7587.,7955.,8324.,8695.,9066.,9438.,9811.,10184.,10557.,10931.])
    BaseLogPartFctRef[215]['NbLabels']=2
    BaseLogPartFctRef[215]['NbCliques']=7496
    BaseLogPartFctRef[215]['NbSites']=2926
    BaseLogPartFctRef[215]['StdNgbhDivMoyNgbh']=0.1825
    BaseLogPartFctRef[216]={}
    BaseLogPartFctRef[216]['LogPF']=np.array([2044.,2234.,2429.,2628.,2832.,3042.,3257.,3478.,3706.,3948.,4211.,4497.,4803.,5123.,5457.,5801.,6152.,6509.,6870.,7234.,7600.,7969.,8338.,8709.,9080.,9452.,9825.,10198.,10572.,10945.])
    BaseLogPartFctRef[216]['NbLabels']=2
    BaseLogPartFctRef[216]['NbCliques']=7504
    BaseLogPartFctRef[216]['NbSites']=2949
    BaseLogPartFctRef[216]['StdNgbhDivMoyNgbh']=0.1890
    BaseLogPartFctRef[217]={}
    BaseLogPartFctRef[217]['LogPF']=np.array([2076.,2271.,2471.,2675.,2885.,3101.,3321.,3549.,3783.,4032.,4306.,4603.,4919.,5250.,5594.,5948.,6310.,6677.,7048.,7423.,7800.,8178.,8558.,8939.,9321.,9703.,10086.,10469.,10853.,11237.])
    BaseLogPartFctRef[217]['NbLabels']=2
    BaseLogPartFctRef[217]['NbCliques']=7703
    BaseLogPartFctRef[217]['NbSites']=2995
    BaseLogPartFctRef[217]['StdNgbhDivMoyNgbh']=0.1787
    BaseLogPartFctRef[218]={}
    BaseLogPartFctRef[218]['LogPF']=np.array([2070.,2264.,2464.,2668.,2877.,3092.,3312.,3538.,3772.,4023.,4298.,4597.,4911.,5243.,5587.,5940.,6301.,6667.,7037.,7411.,7786.,8164.,8543.,8923.,9304.,9685.,10067.,10449.,10832.,11215.])
    BaseLogPartFctRef[218]['NbLabels']=2
    BaseLogPartFctRef[218]['NbCliques']=7683
    BaseLogPartFctRef[218]['NbSites']=2986
    BaseLogPartFctRef[218]['StdNgbhDivMoyNgbh']=0.1788
    BaseLogPartFctRef[219]={}
    BaseLogPartFctRef[219]['LogPF']=np.array([2069.,2263.,2461.,2665.,2873.,3087.,3306.,3531.,3765.,4015.,4283.,4574.,4885.,5214.,5555.,5906.,6265.,6629.,6998.,7369.,7743.,8119.,8496.,8874.,9253.,9633.,10013.,10394.,10775.,11156.])
    BaseLogPartFctRef[219]['NbLabels']=2
    BaseLogPartFctRef[219]['NbCliques']=7650
    BaseLogPartFctRef[219]['NbSites']=2985
    BaseLogPartFctRef[219]['StdNgbhDivMoyNgbh']=0.1807
    BaseLogPartFctRef[220]={}
    BaseLogPartFctRef[220]['LogPF']=np.array([2023.,2210.,2402.,2599.,2801.,3008.,3220.,3439.,3664.,3902.,4165.,4447.,4745.,5061.,5389.,5727.,6073.,6423.,6779.,7138.,7499.,7862.,8226.,8592.,8958.,9326.,9693.,10062.,10430.,10799.])
    BaseLogPartFctRef[220]['NbLabels']=2
    BaseLogPartFctRef[220]['NbCliques']=7408
    BaseLogPartFctRef[220]['NbSites']=2918
    BaseLogPartFctRef[220]['StdNgbhDivMoyNgbh']=0.1983
    BaseLogPartFctRef[221]={}
    BaseLogPartFctRef[221]['LogPF']=np.array([2014.,2202.,2393.,2590.,2791.,2998.,3210.,3428.,3654.,3891.,4153.,4437.,4738.,5054.,5383.,5722.,6068.,6419.,6775.,7134.,7496.,7859.,8224.,8589.,8956.,9323.,9691.,10059.,10428.,10796.])
    BaseLogPartFctRef[221]['NbLabels']=2
    BaseLogPartFctRef[221]['NbCliques']=7401
    BaseLogPartFctRef[221]['NbSites']=2906
    BaseLogPartFctRef[221]['StdNgbhDivMoyNgbh']=0.1820
    BaseLogPartFctRef[222]={}
    BaseLogPartFctRef[222]['LogPF']=np.array([1992.,2176.,2365.,2559.,2757.,2961.,3170.,3385.,3607.,3840.,4096.,4371.,4664.,4975.,5298.,5631.,5972.,6318.,6668.,7021.,7376.,7733.,8092.,8452.,8813.,9174.,9536.,9898.,10261.,10624.])
    BaseLogPartFctRef[222]['NbLabels']=2
    BaseLogPartFctRef[222]['NbCliques']=7285
    BaseLogPartFctRef[222]['NbSites']=2874
    BaseLogPartFctRef[222]['StdNgbhDivMoyNgbh']=0.1896
    BaseLogPartFctRef[223]={}
    BaseLogPartFctRef[223]['LogPF']=np.array([2059.,2253.,2451.,2654.,2862.,3075.,3294.,3519.,3752.,3998.,4271.,4563.,4875.,5202.,5543.,5894.,6252.,6616.,6984.,7355.,7728.,8104.,8480.,8858.,9236.,9615.,9995.,10375.,10755.,11136.])
    BaseLogPartFctRef[223]['NbLabels']=2
    BaseLogPartFctRef[223]['NbCliques']=7636
    BaseLogPartFctRef[223]['NbSites']=2971
    BaseLogPartFctRef[223]['StdNgbhDivMoyNgbh']=0.1798
    BaseLogPartFctRef[224]={}
    BaseLogPartFctRef[224]['LogPF']=np.array([2071.,2265.,2463.,2667.,2875.,3089.,3308.,3534.,3767.,4014.,4283.,4578.,4890.,5219.,5560.,5911.,6270.,6635.,7003.,7375.,7749.,8125.,8502.,8880.,9259.,9639.,10019.,10400.,10781.,11163.])
    BaseLogPartFctRef[224]['NbLabels']=2
    BaseLogPartFctRef[224]['NbCliques']=7652
    BaseLogPartFctRef[224]['NbSites']=2988
    BaseLogPartFctRef[224]['StdNgbhDivMoyNgbh']=0.1819
    BaseLogPartFctRef[225]={}
    BaseLogPartFctRef[225]['LogPF']=np.array([2053.,2245.,2442.,2644.,2851.,3064.,3281.,3506.,3737.,3983.,4254.,4545.,4855.,5182.,5521.,5870.,6227.,6588.,6954.,7324.,7695.,8068.,8443.,8818.,9195.,9572.,9949.,10327.,10706.,11085.])
    BaseLogPartFctRef[225]['NbLabels']=2
    BaseLogPartFctRef[225]['NbCliques']=7599
    BaseLogPartFctRef[225]['NbSites']=2962
    BaseLogPartFctRef[225]['StdNgbhDivMoyNgbh']=0.1821
    BaseLogPartFctRef[226]={}
    BaseLogPartFctRef[226]['LogPF']=np.array([2075.,2269.,2468.,2673.,2882.,3097.,3317.,3543.,3778.,4029.,4302.,4597.,4909.,5239.,5581.,5934.,6295.,6661.,7032.,7405.,7781.,8158.,8537.,8917.,9298.,9680.,10062.,10444.,10827.,11210.])
    BaseLogPartFctRef[226]['NbLabels']=2
    BaseLogPartFctRef[226]['NbCliques']=7685
    BaseLogPartFctRef[226]['NbSites']=2993
    BaseLogPartFctRef[226]['StdNgbhDivMoyNgbh']=0.1803
    BaseLogPartFctRef[227]={}
    BaseLogPartFctRef[227]['LogPF']=np.array([2066.,2260.,2459.,2663.,2872.,3086.,3306.,3532.,3766.,4014.,4288.,4582.,4896.,5227.,5570.,5923.,6283.,6649.,7018.,7391.,7766.,8143.,8522.,8901.,9281.,9662.,10044.,10425.,10808.,11190.])
    BaseLogPartFctRef[227]['NbLabels']=2
    BaseLogPartFctRef[227]['NbCliques']=7673
    BaseLogPartFctRef[227]['NbSites']=2980
    BaseLogPartFctRef[227]['StdNgbhDivMoyNgbh']=0.1807
    BaseLogPartFctRef[228]={}
    BaseLogPartFctRef[228]['LogPF']=np.array([2075.,2270.,2470.,2676.,2886.,3101.,3323.,3550.,3786.,4041.,4316.,4614.,4930.,5261.,5607.,5963.,6325.,6693.,7065.,7440.,7818.,8197.,8577.,8959.,9342.,9725.,10108.,10492.,10877.,11262.])
    BaseLogPartFctRef[228]['NbLabels']=2
    BaseLogPartFctRef[228]['NbCliques']=7718
    BaseLogPartFctRef[228]['NbSites']=2993
    BaseLogPartFctRef[228]['StdNgbhDivMoyNgbh']=0.1810
    BaseLogPartFctRef[229]={}
    BaseLogPartFctRef[229]['LogPF']=np.array([2049.,2240.,2436.,2637.,2843.,3055.,3271.,3494.,3725.,3971.,4239.,4529.,4835.,5158.,5495.,5841.,6195.,6555.,6919.,7286.,7655.,8026.,8399.,8773.,9147.,9522.,9898.,10274.,10651.,11027.])
    BaseLogPartFctRef[229]['NbLabels']=2
    BaseLogPartFctRef[229]['NbCliques']=7560
    BaseLogPartFctRef[229]['NbSites']=2956
    BaseLogPartFctRef[229]['StdNgbhDivMoyNgbh']=0.1824
    BaseLogPartFctRef[230]={}
    BaseLogPartFctRef[230]['LogPF']=np.array([2041.,2231.,2427.,2628.,2834.,3044.,3261.,3483.,3713.,3955.,4226.,4512.,4821.,5146.,5483.,5830.,6184.,6544.,6907.,7274.,7643.,8014.,8386.,8759.,9133.,9508.,9883.,10259.,10635.,11011.])
    BaseLogPartFctRef[230]['NbLabels']=2
    BaseLogPartFctRef[230]['NbCliques']=7546
    BaseLogPartFctRef[230]['NbSites']=2944
    BaseLogPartFctRef[230]['StdNgbhDivMoyNgbh']=0.1754
    BaseLogPartFctRef[231]={}
    BaseLogPartFctRef[231]['LogPF']=np.array([2075.,2270.,2470.,2676.,2886.,3102.,3323.,3550.,3785.,4036.,4313.,4612.,4929.,5262.,5608.,5964.,6327.,6695.,7067.,7442.,7820.,8199.,8580.,8962.,9344.,9727.,10111.,10495.,10880.,11265.])
    BaseLogPartFctRef[231]['NbLabels']=2
    BaseLogPartFctRef[231]['NbCliques']=7720
    BaseLogPartFctRef[231]['NbSites']=2993
    BaseLogPartFctRef[231]['StdNgbhDivMoyNgbh']=0.1826
    BaseLogPartFctRef[232]={}
    BaseLogPartFctRef[232]['LogPF']=np.array([2065.,2258.,2456.,2659.,2867.,3080.,3299.,3524.,3757.,4002.,4273.,4564.,4875.,5201.,5542.,5893.,6250.,6614.,6981.,7352.,7725.,8100.,8476.,8853.,9231.,9610.,9990.,10370.,10750.,11130.])
    BaseLogPartFctRef[232]['NbLabels']=2
    BaseLogPartFctRef[232]['NbCliques']=7633
    BaseLogPartFctRef[232]['NbSites']=2979
    BaseLogPartFctRef[232]['StdNgbhDivMoyNgbh']=0.1800
    BaseLogPartFctRef[233]={}
    BaseLogPartFctRef[233]['LogPF']=np.array([2031.,2219.,2412.,2610.,2813.,3021.,3235.,3454.,3680.,3920.,4180.,4460.,4760.,5076.,5407.,5748.,6096.,6449.,6806.,7167.,7531.,7896.,8262.,8630.,8998.,9368.,9737.,10108.,10478.,10849.])
    BaseLogPartFctRef[233]['NbLabels']=2
    BaseLogPartFctRef[233]['NbCliques']=7442
    BaseLogPartFctRef[233]['NbSites']=2930
    BaseLogPartFctRef[233]['StdNgbhDivMoyNgbh']=0.1851
    BaseLogPartFctRef[234]={}
    BaseLogPartFctRef[234]['LogPF']=np.array([2073.,2268.,2468.,2673.,2883.,3098.,3320.,3547.,3782.,4035.,4313.,4610.,4925.,5257.,5602.,5958.,6320.,6688.,7060.,7435.,7813.,8192.,8572.,8954.,9336.,9719.,10103.,10486.,10871.,11255.])
    BaseLogPartFctRef[234]['NbLabels']=2
    BaseLogPartFctRef[234]['NbCliques']=7713
    BaseLogPartFctRef[234]['NbSites']=2990
    BaseLogPartFctRef[234]['StdNgbhDivMoyNgbh']=0.1761
    BaseLogPartFctRef[235]={}
    BaseLogPartFctRef[235]['LogPF']=np.array([2075.,2269.,2469.,2673.,2882.,3097.,3318.,3544.,3779.,4025.,4296.,4593.,4905.,5235.,5579.,5932.,6292.,6659.,7030.,7403.,7779.,8157.,8536.,8916.,9297.,9679.,10061.,10444.,10826.,11210.])
    BaseLogPartFctRef[235]['NbLabels']=2
    BaseLogPartFctRef[235]['NbCliques']=7686
    BaseLogPartFctRef[235]['NbSites']=2993
    BaseLogPartFctRef[235]['StdNgbhDivMoyNgbh']=0.1758
    BaseLogPartFctRef[236]={}
    BaseLogPartFctRef[236]['LogPF']=np.array([2070.,2266.,2466.,2672.,2883.,3099.,3321.,3550.,3786.,4039.,4316.,4614.,4931.,5266.,5613.,5971.,6335.,6705.,7079.,7455.,7834.,8215.,8597.,8980.,9363.,9748.,10133.,10518.,10904.,11289.])
    BaseLogPartFctRef[236]['NbLabels']=2
    BaseLogPartFctRef[236]['NbCliques']=7741
    BaseLogPartFctRef[236]['NbSites']=2986
    BaseLogPartFctRef[236]['StdNgbhDivMoyNgbh']=0.1784
    BaseLogPartFctRef[237]={}
    BaseLogPartFctRef[237]['LogPF']=np.array([2054.,2246.,2443.,2644.,2850.,3062.,3279.,3502.,3733.,3979.,4247.,4536.,4842.,5165.,5503.,5850.,6204.,6564.,6929.,7296.,7666.,8037.,8410.,8785.,9160.,9535.,9911.,10288.,10665.,11042.])
    BaseLogPartFctRef[237]['NbLabels']=2
    BaseLogPartFctRef[237]['NbCliques']=7569
    BaseLogPartFctRef[237]['NbSites']=2964
    BaseLogPartFctRef[237]['StdNgbhDivMoyNgbh']=0.1798
    BaseLogPartFctRef[238]={}
    BaseLogPartFctRef[238]['LogPF']=np.array([2076.,2272.,2472.,2678.,2889.,3105.,3326.,3554.,3790.,4044.,4318.,4618.,4933.,5266.,5612.,5968.,6331.,6700.,7073.,7449.,7828.,8208.,8589.,8972.,9355.,9739.,10124.,10509.,10894.,11279.])
    BaseLogPartFctRef[238]['NbLabels']=2
    BaseLogPartFctRef[238]['NbCliques']=7732
    BaseLogPartFctRef[238]['NbSites']=2995
    BaseLogPartFctRef[238]['StdNgbhDivMoyNgbh']=0.1768
    BaseLogPartFctRef[239]={}
    BaseLogPartFctRef[239]['LogPF']=np.array([2066.,2261.,2460.,2664.,2873.,3088.,3308.,3535.,3769.,4019.,4292.,4586.,4898.,5229.,5573.,5926.,6287.,6652.,7022.,7395.,7771.,8148.,8526.,8906.,9286.,9668.,10049.,10431.,10814.,11197.])
    BaseLogPartFctRef[239]['NbLabels']=2
    BaseLogPartFctRef[239]['NbCliques']=7679
    BaseLogPartFctRef[239]['NbSites']=2981
    BaseLogPartFctRef[239]['StdNgbhDivMoyNgbh']=0.1819



    #regular grids: lines - squares - cubes

    BaseLogPartFctRef[240]={}
    BaseLogPartFctRef[240]['LogPF']=np.array([693.1,718.2,743.9,770.3,797.6,825.2,853.6,882.7,912.4,942.8,973.6,1005.,1037.,1070.,1103.,1137.,1171.,1206.,1241.,1277.,1314.,1351.,1388.,1426.,1464.,1503.,1543.,1583.,1623.,1663.])
    BaseLogPartFctRef[240]['NbLabels']=2
    BaseLogPartFctRef[240]['NbCliques']=999
    BaseLogPartFctRef[240]['NbSites']=1000
    BaseLogPartFctRef[240]['StdNgbhDivMoyNgbh']=0.0223
    BaseLogPartFctRef[241]={}
    BaseLogPartFctRef[241]['LogPF']=np.array([693.1,761.6,831.6,903.3,977.0,1052.,1130.,1211.,1295.,1386.,1485.,1592.,1708.,1828.,1953.,2080.,2209.,2340.,2471.,2603.,2736.,2870.,3004.,3138.,3272.,3407.,3541.,3676.,3811.,3946.])
    BaseLogPartFctRef[241]['NbLabels']=2
    BaseLogPartFctRef[241]['NbCliques']=2700
    BaseLogPartFctRef[241]['NbSites']=1000
    BaseLogPartFctRef[241]['StdNgbhDivMoyNgbh']=0.1282
    BaseLogPartFctRef[242]={}
    BaseLogPartFctRef[242]['LogPF']=np.array([1198.,1318.,1441.,1567.,1697.,1830.,1966.,2107.,2257.,2425.,2606.,2805.,3012.,3227.,3449.,3674.,3903.,4134.,4366.,4600.,4835.,5070.,5306.,5543.,5779.,6016.,6253.,6490.,6727.,6964.])
    BaseLogPartFctRef[242]['NbLabels']=2
    BaseLogPartFctRef[242]['NbCliques']=4752
    BaseLogPartFctRef[242]['NbSites']=1728
    BaseLogPartFctRef[242]['StdNgbhDivMoyNgbh']=0.1173
    BaseLogPartFctRef[243]={}
    BaseLogPartFctRef[243]['LogPF']=np.array([10830.,11614.,12417.,13239.,14084.,14952.,15845.,16763.,17713.,18690.,19705.,20756.,21857.,23002.,24195.,25449.,26770.,28140.,29560.,31007.,32472.,33955.,35452.,36961.,38477.,39999.,41528.,43061.,44598.,46135.])
    BaseLogPartFctRef[243]['NbLabels']=2
    BaseLogPartFctRef[243]['NbCliques']=31000
    BaseLogPartFctRef[243]['NbSites']=15625
    BaseLogPartFctRef[243]['StdNgbhDivMoyNgbh']=0.0447
    BaseLogPartFctRef[244]={}
    BaseLogPartFctRef[244]['LogPF']=np.array([2339.,2578.,2822.,3074.,3332.,3596.,3869.,4156.,4468.,4810.,5192.,5597.,6014.,6449.,6894.,7347.,7804.,8265.,8729.,9195.,9663.,10132.,10601.,11071.,11541.,12013.,12484.,12956.,13428.,13900.])
    BaseLogPartFctRef[244]['NbLabels']=2
    BaseLogPartFctRef[244]['NbCliques']=9450
    BaseLogPartFctRef[244]['NbSites']=3375
    BaseLogPartFctRef[244]['StdNgbhDivMoyNgbh']=0.1051
    BaseLogPartFctRef[245]={}
    BaseLogPartFctRef[245]['LogPF']=np.array([10830.,11227.,11632.,12049.,12475.,12913.,13359.,13818.,14285.,14763.,15254.,15754.,16263.,16785.,17315.,17853.,18401.,18959.,19524.,20101.,20686.,21278.,21875.,22482.,23097.,23719.,24348.,24982.,25627.,26275.])
    BaseLogPartFctRef[245]['NbLabels']=2
    BaseLogPartFctRef[245]['NbCliques']=15624
    BaseLogPartFctRef[245]['NbSites']=15625
    BaseLogPartFctRef[245]['StdNgbhDivMoyNgbh']=0.0057
    BaseLogPartFctRef[246]={}
    BaseLogPartFctRef[246]['LogPF']=np.array([1198.,1241.,1286.,1332.,1379.,1427.,1476.,1526.,1578.,1630.,1684.,1739.,1794.,1851.,1908.,1967.,2026.,2086.,2147.,2210.,2273.,2337.,2402.,2467.,2534.,2601.,2669.,2738.,2807.,2877.])
    BaseLogPartFctRef[246]['NbLabels']=2
    BaseLogPartFctRef[246]['NbCliques']=1727
    BaseLogPartFctRef[246]['NbSites']=1728
    BaseLogPartFctRef[246]['StdNgbhDivMoyNgbh']=0.0170
    BaseLogPartFctRef[247]={}
    BaseLogPartFctRef[247]['LogPF']=np.array([5545.,6122.,6715.,7322.,7946.,8585.,9250.,9949.,10720.,11592.,12535.,13531.,14569.,15629.,16710.,17808.,18917.,20035.,21158.,22285.,23414.,24546.,25680.,26815.,27952.,29090.,30228.,31367.,32505.,33645.])
    BaseLogPartFctRef[247]['NbLabels']=2
    BaseLogPartFctRef[247]['NbCliques']=22800
    BaseLogPartFctRef[247]['NbSites']=8000
    BaseLogPartFctRef[247]['StdNgbhDivMoyNgbh']=0.0911
    BaseLogPartFctRef[248]={}
    BaseLogPartFctRef[248]['LogPF']=np.array([10830.,11968.,13133.,14329.,15560.,16832.,18149.,19549.,21106.,22849.,24742.,26729.,28789.,30902.,33054.,35232.,37428.,39637.,41856.,44085.,46317.,48554.,50795.,53037.,55282.,57528.,59775.,62022.,64270.,66519.])
    BaseLogPartFctRef[248]['NbLabels']=2
    BaseLogPartFctRef[248]['NbCliques']=45000
    BaseLogPartFctRef[248]['NbSites']=15625
    BaseLogPartFctRef[248]['StdNgbhDivMoyNgbh']=0.0816
    BaseLogPartFctRef[249]={}
    BaseLogPartFctRef[249]['LogPF']=np.array([666.1,713.0,761.3,810.8,861.5,913.5,966.7,1021.,1077.,1134.,1193.,1254.,1317.,1382.,1449.,1518.,1591.,1667.,1746.,1828.,1914.,2000.,2088.,2177.,2266.,2357.,2447.,2539.,2631.,2723.])
    BaseLogPartFctRef[249]['NbLabels']=2
    BaseLogPartFctRef[249]['NbCliques']=1860
    BaseLogPartFctRef[249]['NbSites']=961
    BaseLogPartFctRef[249]['StdNgbhDivMoyNgbh']=0.0897
    BaseLogPartFctRef[250]={}
    BaseLogPartFctRef[250]['LogPF']=np.array([2339.,2425.,2512.,2603.,2694.,2788.,2884.,2982.,3083.,3185.,3290.,3396.,3505.,3616.,3729.,3843.,3960.,4079.,4199.,4320.,4444.,4570.,4698.,4828.,4958.,5089.,5223.,5357.,5494.,5631.])
    BaseLogPartFctRef[250]['NbLabels']=2
    BaseLogPartFctRef[250]['NbCliques']=3374
    BaseLogPartFctRef[250]['NbSites']=3375
    BaseLogPartFctRef[250]['StdNgbhDivMoyNgbh']=0.0122
    BaseLogPartFctRef[251]={}
    BaseLogPartFctRef[251]['LogPF']=np.array([1165.,1248.,1333.,1420.,1509.,1601.,1695.,1791.,1890.,1991.,2095.,2203.,2314.,2429.,2548.,2674.,2806.,2943.,3086.,3234.,3385.,3539.,3695.,3853.,4012.,4172.,4333.,4494.,4656.,4818.])
    BaseLogPartFctRef[251]['NbLabels']=2
    BaseLogPartFctRef[251]['NbCliques']=3280
    BaseLogPartFctRef[251]['NbSites']=1681
    BaseLogPartFctRef[251]['StdNgbhDivMoyNgbh']=0.0780
    BaseLogPartFctRef[252]={}
    BaseLogPartFctRef[252]['LogPF']=np.array([2332.,2499.,2671.,2846.,3026.,3210.,3400.,3594.,3794.,4000.,4213.,4433.,4661.,4895.,5142.,5403.,5674.,5955.,6250.,6553.,6861.,7173.,7491.,7811.,8133.,8456.,8781.,9107.,9434.,9761.])
    BaseLogPartFctRef[252]['NbLabels']=2
    BaseLogPartFctRef[252]['NbCliques']=6612
    BaseLogPartFctRef[252]['NbSites']=3364
    BaseLogPartFctRef[252]['StdNgbhDivMoyNgbh']=0.0656
    BaseLogPartFctRef[253]={}
    BaseLogPartFctRef[253]['LogPF']=np.array([5545.,5747.,5956.,6168.,6386.,6609.,6837.,7069.,7307.,7551.,7799.,8053.,8311.,8575.,8844.,9117.,9394.,9677.,9964.,10256.,10552.,10852.,11156.,11463.,11775.,12090.,12409.,12730.,13056.,13384.])
    BaseLogPartFctRef[253]['NbLabels']=2
    BaseLogPartFctRef[253]['NbCliques']=7999
    BaseLogPartFctRef[253]['NbSites']=8000
    BaseLogPartFctRef[253]['StdNgbhDivMoyNgbh']=0.0079
    BaseLogPartFctRef[254]={}
    BaseLogPartFctRef[254]['LogPF']=np.array([5490.,5888.,6294.,6711.,7139.,7577.,8028.,8489.,8964.,9455.,9963.,10495.,11038.,11610.,12207.,12837.,13496.,14183.,14893.,15616.,16352.,17099.,17852.,18611.,19376.,20143.,20914.,21689.,22464.,23241.])
    BaseLogPartFctRef[254]['NbLabels']=2
    BaseLogPartFctRef[254]['NbCliques']=15664
    BaseLogPartFctRef[254]['NbSites']=7921
    BaseLogPartFctRef[254]['StdNgbhDivMoyNgbh']=0.0530


    #ComputeBaseLogPartFctRef_NonReg(FirstIndex=(255+0),NbExtraIndex=30,BetaMax=1.45,DeltaBeta=0.05,NbLabels=3,NBX=10,NBY=10,NBZ=10,BetaGeneration=0.2)
    BaseLogPartFctRef[255 ]={}
    BaseLogPartFctRef[255]['LogPF']=np.array([71.41,73.01,74.72,76.50,78.23,80.06,81.98,84.04,86.13,88.26,90.30,92.47,94.95,97.21,99.74,102.3,104.9,107.9,111.0,114.0,117.1,120.1,123.3,126.7,130.4,134.2,138.0,142.0,146.0,150.2])
    BaseLogPartFctRef[ 255 ]['NbLabels']= 3
    BaseLogPartFctRef[ 255 ]['NbCliques']= 96
    BaseLogPartFctRef[ 255 ]['NbSites']= 65
    BaseLogPartFctRef[ 255 ]['StdNgbhDivMoyNgbh']= 0.357726769776
    BaseLogPartFctRef[ 256 ]={}
    BaseLogPartFctRef[256]['LogPF']=np.array([32.96,33.70,34.47,35.25,36.05,36.87,37.72,38.60,39.52,40.50,41.49,42.56,43.62,44.72,45.88,47.08,48.31,49.59,50.87,52.31,53.77,55.33,56.95,58.62,60.30,61.95,63.72,65.55,67.39,69.26])
    BaseLogPartFctRef[ 256 ]['NbLabels']= 3
    BaseLogPartFctRef[ 256 ]['NbCliques']= 43
    BaseLogPartFctRef[ 256 ]['NbSites']= 30
    BaseLogPartFctRef[ 256 ]['StdNgbhDivMoyNgbh']= 0.405898164708
    BaseLogPartFctRef[ 257 ]={}
    BaseLogPartFctRef[257]['LogPF']=np.array([24.17,24.63,25.11,25.59,26.13,26.67,27.22,27.82,28.42,29.01,29.68,30.34,31.00,31.67,32.40,33.15,33.94,34.74,35.59,36.46,37.35,38.27,39.19,40.20,41.17,42.22,43.29,44.34,45.43,46.60])
    BaseLogPartFctRef[ 257 ]['NbLabels']= 3
    BaseLogPartFctRef[ 257 ]['NbCliques']= 28
    BaseLogPartFctRef[ 257 ]['NbSites']= 22
    BaseLogPartFctRef[ 257 ]['StdNgbhDivMoyNgbh']= 0.352639105663
    BaseLogPartFctRef[ 258 ]={}
    BaseLogPartFctRef[258]['LogPF']=np.array([18.68,19.04,19.41,19.78,20.18,20.62,21.08,21.55,22.02,22.52,23.01,23.52,24.07,24.58,25.15,25.78,26.40,27.05,27.69,28.38,29.06,29.80,30.55,31.31,32.10,32.93,33.78,34.68,35.60,36.48])
    BaseLogPartFctRef[ 258 ]['NbLabels']= 3
    BaseLogPartFctRef[ 258 ]['NbCliques']= 22
    BaseLogPartFctRef[ 258 ]['NbSites']= 17
    BaseLogPartFctRef[ 258 ]['StdNgbhDivMoyNgbh']= 0.414774747159
    BaseLogPartFctRef[ 259 ]={}
    BaseLogPartFctRef[259]['LogPF']=np.array([28.56,29.17,29.80,30.44,31.12,31.84,32.54,33.30,34.07,34.86,35.71,36.54,37.37,38.28,39.24,40.22,41.27,42.36,43.51,44.69,45.91,47.20,48.49,49.81,51.21,52.62,54.08,55.63,57.09,58.60])
    BaseLogPartFctRef[ 259 ]['NbLabels']= 3
    BaseLogPartFctRef[ 259 ]['NbCliques']= 36
    BaseLogPartFctRef[ 259 ]['NbSites']= 26
    BaseLogPartFctRef[ 259 ]['StdNgbhDivMoyNgbh']= 0.433976410504
    BaseLogPartFctRef[ 260 ]={}
    BaseLogPartFctRef[260]['LogPF']=np.array([40.65,41.50,42.37,43.26,44.18,45.13,46.15,47.21,48.26,49.41,50.52,51.70,52.89,54.14,55.44,56.79,58.17,59.56,61.13,62.71,64.21,65.97,67.69,69.47,71.31,73.24,75.22,77.21,79.28,81.33])
    BaseLogPartFctRef[ 260 ]['NbLabels']= 3
    BaseLogPartFctRef[ 260 ]['NbCliques']= 50
    BaseLogPartFctRef[ 260 ]['NbSites']= 37
    BaseLogPartFctRef[ 260 ]['StdNgbhDivMoyNgbh']= 0.400221998294
    BaseLogPartFctRef[261 ]={}
    BaseLogPartFctRef[261]['LogPF']=np.array([29.66,30.27,30.92,31.58,32.25,32.96,33.71,34.48,35.25,36.09,36.92,37.84,38.75,39.69,40.66,41.64,42.68,43.76,44.86,46.07,47.29,48.55,49.86,51.17,52.56,54.01,55.47,56.95,58.54,60.07])
    BaseLogPartFctRef[ 261 ]['NbLabels']= 3
    BaseLogPartFctRef[ 261 ]['NbCliques']= 37
    BaseLogPartFctRef[ 261 ]['NbSites']= 27
    BaseLogPartFctRef[ 261 ]['StdNgbhDivMoyNgbh']= 0.401087998191
    BaseLogPartFctRef[262 ]={}
    BaseLogPartFctRef[262]['LogPF']=np.array([42.85,43.63,44.46,45.36,46.28,47.20,48.17,49.18,50.21,51.25,52.33,53.43,54.61,55.82,57.10,58.39,59.74,61.03,62.41,63.81,65.31,66.85,68.44,70.07,71.76,73.44,75.17,76.89,78.64,80.45])
    BaseLogPartFctRef[ 262 ]['NbLabels']= 3
    BaseLogPartFctRef[ 262 ]['NbCliques']= 48
    BaseLogPartFctRef[ 262 ]['NbSites']= 39
    BaseLogPartFctRef[ 262 ]['StdNgbhDivMoyNgbh']= 0.367913258811
    BaseLogPartFctRef[263 ]={}
    BaseLogPartFctRef[263]['LogPF']=np.array([37.35,38.18,39.02,39.88,40.77,41.70,42.68,43.68,44.71,45.78,46.92,48.05,49.21,50.47,51.76,53.06,54.47,55.90,57.39,59.03,60.57,62.21,63.98,65.79,67.55,69.39,71.34,73.24,75.26,77.30])
    BaseLogPartFctRef[ 263 ]['NbLabels']= 3
    BaseLogPartFctRef[ 263 ]['NbCliques']= 48
    BaseLogPartFctRef[ 263 ]['NbSites']= 34
    BaseLogPartFctRef[ 263 ]['StdNgbhDivMoyNgbh']= 0.367881359164
    BaseLogPartFctRef[264 ]={}
    BaseLogPartFctRef[264]['LogPF']=np.array([38.45,39.27,40.13,41.00,41.87,42.84,43.86,44.86,45.91,47.01,48.13,49.30,50.51,51.79,53.04,54.41,55.81,57.23,58.77,60.41,62.10,63.77,65.56,67.33,69.16,71.09,73.08,75.09,77.19,79.23])
    BaseLogPartFctRef[ 264 ]['NbLabels']= 3
    BaseLogPartFctRef[ 264 ]['NbCliques']= 49
    BaseLogPartFctRef[ 264 ]['NbSites']= 35
    BaseLogPartFctRef[ 264 ]['StdNgbhDivMoyNgbh']= 0.437276577262
    BaseLogPartFctRef[265 ]={}
    BaseLogPartFctRef[265]['LogPF']=np.array([25.27,25.70,26.15,26.63,27.10,27.60,28.12,28.64,29.18,29.75,30.32,30.92,31.57,32.23,32.88,33.60,34.30,35.01,35.79,36.58,37.32,38.14,38.98,39.83,40.65,41.53,42.45,43.38,44.37,45.29])
    BaseLogPartFctRef[ 265 ]['NbLabels']= 3
    BaseLogPartFctRef[ 265 ]['NbCliques']= 26
    BaseLogPartFctRef[ 265 ]['NbSites']= 23
    BaseLogPartFctRef[ 265 ]['StdNgbhDivMoyNgbh']= 0.417846099022
    BaseLogPartFctRef[266 ]={}
    BaseLogPartFctRef[266]['LogPF']=np.array([30.76,31.43,32.09,32.77,33.51,34.27,35.04,35.82,36.63,37.47,38.28,39.17,40.10,41.08,42.07,43.13,44.12,45.24,46.37,47.57,48.83,50.09,51.39,52.74,54.17,55.67,57.14,58.70,60.24,61.78])
    BaseLogPartFctRef[ 266 ]['NbLabels']= 3
    BaseLogPartFctRef[ 266 ]['NbCliques']= 38
    BaseLogPartFctRef[ 266 ]['NbSites']= 28
    BaseLogPartFctRef[ 266 ]['StdNgbhDivMoyNgbh']= 0.412310219784
    BaseLogPartFctRef[267 ]={}
    BaseLogPartFctRef[267]['LogPF']=np.array([26.37,26.93,27.53,28.17,28.81,29.47,30.12,30.84,31.56,32.31,33.04,33.86,34.71,35.55,36.45,37.40,38.33,39.35,40.37,41.47,42.58,43.75,45.01,46.26,47.52,48.83,50.16,51.57,52.99,54.45])
    BaseLogPartFctRef[ 267 ]['NbLabels']= 3
    BaseLogPartFctRef[ 267 ]['NbCliques']= 34
    BaseLogPartFctRef[ 267 ]['NbSites']= 24
    BaseLogPartFctRef[ 267 ]['StdNgbhDivMoyNgbh']= 0.34750373056
    BaseLogPartFctRef[268 ]={}
    BaseLogPartFctRef[268]['LogPF']=np.array([23.07,23.53,23.99,24.44,24.93,25.42,25.92,26.44,26.98,27.53,28.13,28.76,29.39,30.07,30.73,31.41,32.10,32.88,33.62,34.38,35.23,36.05,36.90,37.82,38.73,39.66,40.66,41.67,42.68,43.68])
    BaseLogPartFctRef[ 268 ]['NbLabels']= 3
    BaseLogPartFctRef[ 268 ]['NbCliques']= 26
    BaseLogPartFctRef[ 268 ]['NbSites']= 21
    BaseLogPartFctRef[ 268 ]['StdNgbhDivMoyNgbh']= 0.348466988836
    BaseLogPartFctRef[269 ]={}
    BaseLogPartFctRef[269]['LogPF']=np.array([36.25,37.01,37.78,38.56,39.37,40.22,41.11,42.02,42.97,43.99,45.00,46.02,47.11,48.19,49.35,50.50,51.67,52.91,54.27,55.67,57.06,58.50,59.98,61.52,63.14,64.79,66.47,68.15,69.86,71.66])
    BaseLogPartFctRef[ 269 ]['NbLabels']= 3
    BaseLogPartFctRef[ 269 ]['NbCliques']= 44
    BaseLogPartFctRef[ 269 ]['NbSites']= 33
    BaseLogPartFctRef[ 269 ]['StdNgbhDivMoyNgbh']= 0.416697970274
    BaseLogPartFctRef[270 ]={}
    BaseLogPartFctRef[270]['LogPF']=np.array([35.16,35.90,36.66,37.41,38.23,39.07,39.95,40.85,41.75,42.75,43.71,44.75,45.82,46.94,48.06,49.26,50.47,51.75,53.08,54.46,55.90,57.39,58.92,60.48,62.14,63.77,65.54,67.27,69.03,70.90])
    BaseLogPartFctRef[ 270 ]['NbLabels']= 3
    BaseLogPartFctRef[ 270 ]['NbCliques']= 44
    BaseLogPartFctRef[ 270 ]['NbSites']= 32
    BaseLogPartFctRef[ 270 ]['StdNgbhDivMoyNgbh']= 0.32952096304
    BaseLogPartFctRef[271 ]={}
    BaseLogPartFctRef[271]['LogPF']=np.array([24.17,24.63,25.12,25.61,26.10,26.62,27.18,27.74,28.33,28.96,29.62,30.32,31.01,31.71,32.46,33.23,33.99,34.81,35.64,36.44,37.32,38.26,39.19,40.18,41.21,42.21,43.31,44.40,45.48,46.59])
    BaseLogPartFctRef[ 271 ]['NbLabels']= 3
    BaseLogPartFctRef[ 271 ]['NbCliques']= 28
    BaseLogPartFctRef[ 271 ]['NbSites']= 22
    BaseLogPartFctRef[ 271 ]['StdNgbhDivMoyNgbh']= 0.387198296305
    BaseLogPartFctRef[272 ]={}
    BaseLogPartFctRef[272]['LogPF']=np.array([31.86,32.54,33.22,33.93,34.67,35.44,36.24,37.08,37.90,38.74,39.64,40.56,41.47,42.44,43.50,44.60,45.70,46.82,48.02,49.23,50.53,51.86,53.26,54.70,56.12,57.62,59.06,60.57,62.12,63.79])
    BaseLogPartFctRef[ 272 ]['NbLabels']= 3
    BaseLogPartFctRef[ 272 ]['NbCliques']= 39
    BaseLogPartFctRef[ 272 ]['NbSites']= 29
    BaseLogPartFctRef[ 272 ]['StdNgbhDivMoyNgbh']= 0.354031644642
    BaseLogPartFctRef[273 ]={}
    BaseLogPartFctRef[273]['LogPF']=np.array([54.93,56.03,57.20,58.37,59.63,60.90,62.29,63.57,64.88,66.47,68.00,69.60,71.18,72.81,74.67,76.56,78.53,80.31,82.27,84.29,86.25,88.65,90.88,93.13,95.45,97.80,100.3,102.8,105.6,108.1])
    BaseLogPartFctRef[ 273 ]['NbLabels']= 3
    BaseLogPartFctRef[ 273 ]['NbCliques']= 66
    BaseLogPartFctRef[ 273 ]['NbSites']= 50
    BaseLogPartFctRef[ 273 ]['StdNgbhDivMoyNgbh']= 0.361888983479
    BaseLogPartFctRef[274 ]={}
    BaseLogPartFctRef[274]['LogPF']=np.array([27.47,28.01,28.58,29.16,29.78,30.42,31.06,31.72,32.43,33.17,33.92,34.68,35.49,36.31,37.19,38.06,39.03,40.03,41.00,42.04,43.06,44.15,45.23,46.40,47.66,48.96,50.28,51.63,52.94,54.31])
    BaseLogPartFctRef[ 274 ]['NbLabels']= 3
    BaseLogPartFctRef[ 274 ]['NbCliques']= 33
    BaseLogPartFctRef[ 274 ]['NbSites']= 25
    BaseLogPartFctRef[ 274 ]['StdNgbhDivMoyNgbh']= 0.383183705377
    BaseLogPartFctRef[275 ]={}
    BaseLogPartFctRef[275]['LogPF']=np.array([46.14,47.08,48.05,49.03,50.02,51.06,52.20,53.34,54.49,55.73,56.92,58.22,59.53,60.95,62.37,63.82,65.39,66.98,68.60,70.35,72.06,73.84,75.68,77.53,79.47,81.40,83.46,85.55,87.65,89.83])
    BaseLogPartFctRef[ 275 ]['NbLabels']= 3
    BaseLogPartFctRef[ 275 ]['NbCliques']= 55
    BaseLogPartFctRef[ 275 ]['NbSites']= 42
    BaseLogPartFctRef[ 275 ]['StdNgbhDivMoyNgbh']= 0.398066719122
    BaseLogPartFctRef[276 ]={}
    BaseLogPartFctRef[276]['LogPF']=np.array([40.65,41.41,42.22,43.07,43.92,44.78,45.66,46.57,47.49,48.50,49.52,50.60,51.69,52.75,53.95,55.12,56.35,57.64,58.98,60.33,61.73,63.13,64.58,66.08,67.61,69.15,70.72,72.40,74.08,75.80])
    BaseLogPartFctRef[ 276 ]['NbLabels']= 3
    BaseLogPartFctRef[ 276 ]['NbCliques']= 45
    BaseLogPartFctRef[ 276 ]['NbSites']= 37
    BaseLogPartFctRef[ 276 ]['StdNgbhDivMoyNgbh']= 0.411095207391
    BaseLogPartFctRef[277 ]={}
    BaseLogPartFctRef[277]['LogPF']=np.array([24.17,24.63,25.11,25.60,26.09,26.60,27.15,27.69,28.26,28.84,29.47,30.09,30.77,31.43,32.14,32.87,33.62,34.37,35.14,36.03,36.84,37.68,38.58,39.47,40.42,41.41,42.45,43.47,44.55,45.62])
    BaseLogPartFctRef[ 277 ]['NbLabels']= 3
    BaseLogPartFctRef[ 277 ]['NbCliques']= 27
    BaseLogPartFctRef[ 277 ]['NbSites']= 22
    BaseLogPartFctRef[ 277 ]['StdNgbhDivMoyNgbh']= 0.384037694133
    BaseLogPartFctRef[278 ]={}
    BaseLogPartFctRef[278]['LogPF']=np.array([26.37,26.94,27.53,28.15,28.80,29.45,30.13,30.84,31.56,32.26,33.01,33.83,34.69,35.57,36.52,37.49,38.48,39.45,40.53,41.64,42.78,43.98,45.17,46.43,47.75,49.14,50.50,51.94,53.37,54.83])
    BaseLogPartFctRef[ 278 ]['NbLabels']= 3
    BaseLogPartFctRef[ 278 ]['NbCliques']= 34
    BaseLogPartFctRef[ 278 ]['NbSites']= 24
    BaseLogPartFctRef[ 278 ]['StdNgbhDivMoyNgbh']= 0.318891293476
    BaseLogPartFctRef[279 ]={}
    BaseLogPartFctRef[279]['LogPF']=np.array([23.07,23.58,24.09,24.62,25.15,25.75,26.32,26.92,27.52,28.16,28.81,29.51,30.22,31.00,31.76,32.55,33.38,34.21,35.12,35.98,36.96,37.93,38.92,39.94,41.01,42.03,43.13,44.27,45.47,46.67])
    BaseLogPartFctRef[ 279 ]['NbLabels']= 3
    BaseLogPartFctRef[ 279 ]['NbCliques']= 29
    BaseLogPartFctRef[ 279 ]['NbSites']= 21
    BaseLogPartFctRef[ 279 ]['StdNgbhDivMoyNgbh']= 0.382280680684
    BaseLogPartFctRef[280 ]={}
    BaseLogPartFctRef[280]['LogPF']=np.array([32.96,33.53,34.13,34.73,35.36,36.02,36.69,37.40,38.11,38.82,39.59,40.40,41.22,42.08,43.00,43.90,44.79,45.76,46.76,47.77,48.83,49.91,50.98,52.11,53.29,54.48,55.73,56.90,58.16,59.46])
    BaseLogPartFctRef[ 280 ]['NbLabels']= 3
    BaseLogPartFctRef[ 280 ]['NbCliques']= 34
    BaseLogPartFctRef[ 280 ]['NbSites']= 30
    BaseLogPartFctRef[ 280 ]['StdNgbhDivMoyNgbh']= 0.46630918092
    BaseLogPartFctRef[281 ]={}
    BaseLogPartFctRef[281]['LogPF']=np.array([37.35,38.11,38.92,39.74,40.61,41.50,42.42,43.35,44.36,45.37,46.44,47.47,48.63,49.78,50.99,52.29,53.56,54.96,56.33,57.82,59.44,61.03,62.69,64.38,66.12,67.90,69.71,71.51,73.48,75.41])
    BaseLogPartFctRef[ 281 ]['NbLabels']= 3
    BaseLogPartFctRef[ 281 ]['NbCliques']= 46
    BaseLogPartFctRef[ 281 ]['NbSites']= 34
    BaseLogPartFctRef[ 281 ]['StdNgbhDivMoyNgbh']= 0.456457258658
    BaseLogPartFctRef[282 ]={}
    BaseLogPartFctRef[282]['LogPF']=np.array([35.16,35.87,36.61,37.39,38.20,39.01,39.88,40.80,41.73,42.67,43.63,44.64,45.73,46.82,48.01,49.14,50.38,51.68,53.10,54.45,55.85,57.28,58.81,60.36,62.01,63.72,65.40,67.16,68.90,70.67])
    BaseLogPartFctRef[ 282 ]['NbLabels']= 3
    BaseLogPartFctRef[ 282 ]['NbCliques']= 43
    BaseLogPartFctRef[ 282 ]['NbSites']= 32
    BaseLogPartFctRef[ 282 ]['StdNgbhDivMoyNgbh']= 0.480561482005
    BaseLogPartFctRef[283 ]={}
    BaseLogPartFctRef[283]['LogPF']=np.array([50.54,51.68,52.87,54.09,55.33,56.64,58.01,59.43,60.84,62.36,63.89,65.42,67.07,68.75,70.60,72.39,74.33,76.30,78.40,80.55,82.84,85.17,87.65,90.19,92.89,95.61,98.32,101.2,104.0,106.9])
    BaseLogPartFctRef[ 283 ]['NbLabels']= 3
    BaseLogPartFctRef[ 283 ]['NbCliques']= 67
    BaseLogPartFctRef[ 283 ]['NbSites']= 46
    BaseLogPartFctRef[ 283 ]['StdNgbhDivMoyNgbh']= 0.413234085927
    BaseLogPartFctRef[284 ]={}
    BaseLogPartFctRef[284]['LogPF']=np.array([59.33,60.63,61.96,63.41,64.85,66.36,67.91,69.51,71.22,72.97,74.82,76.62,78.46,80.45,82.41,84.47,86.59,88.69,91.15,93.66,96.40,99.02,101.9,104.7,107.8,110.8,113.8,117.0,120.2,123.4])
    BaseLogPartFctRef[ 284 ]['NbLabels']= 3
    BaseLogPartFctRef[ 284 ]['NbCliques']= 78
    BaseLogPartFctRef[ 284 ]['NbSites']= 54
    BaseLogPartFctRef[ 284 ]['StdNgbhDivMoyNgbh']= 0.41179611259

    #In [3]: ComputeBaseLogPartFctRef_NonReg(FirstIndex=(255+30),NbExtraIndex=30,BetaMax=1.45,DeltaBeta=0.05,NbLabels=3,NBX=10,NBY=10,NBZ=10,BetaGeneration=0.4)
    BaseLogPartFctRef[285 ]={}
    BaseLogPartFctRef[285]['LogPF']=np.array([691.0,713.8,737.8,762.5,787.5,814.1,842.4,870.4,899.3,929.5,964.6,1003.7,1042.4,1086.1,1136.3,1186.0,1242.4,1299.1,1359.9,1418.3,1479.8,1542.0,1605.7,1669.4,1734.8,1799.7,1865.2,1931.2,1997.7,2063.7])
    BaseLogPartFctRef[ 285 ]['NbLabels']= 3
    BaseLogPartFctRef[ 285 ]['NbCliques']= 1358
    BaseLogPartFctRef[ 285 ]['NbSites']= 629
    BaseLogPartFctRef[ 285 ]['StdNgbhDivMoyNgbh']= 0.282723872571
    BaseLogPartFctRef[286 ]={}
    BaseLogPartFctRef[286]['LogPF']=np.array([793.2,822.3,851.8,882.8,914.5,947.4,981.4,1017.1,1055.9,1095.6,1145.2,1198.3,1258.3,1323.9,1390.5,1461.3,1533.9,1609.2,1686.6,1765.6,1844.5,1925.2,2006.7,2088.6,2171.4,2253.6,2336.5,2420.1,2503.7,2587.5])
    BaseLogPartFctRef[ 286 ]['NbLabels']= 3
    BaseLogPartFctRef[ 286 ]['NbCliques']= 1697
    BaseLogPartFctRef[ 286 ]['NbSites']= 722
    BaseLogPartFctRef[ 286 ]['StdNgbhDivMoyNgbh']= 0.239718877916
    BaseLogPartFctRef[287 ]={}
    BaseLogPartFctRef[287]['LogPF']=np.array([725.1,750.1,775.7,801.8,829.4,857.8,886.6,917.0,949.6,983.0,1020.9,1063.2,1108.6,1160.3,1216.3,1275.8,1340.0,1402.5,1466.5,1532.7,1600.4,1667.7,1736.5,1806.4,1876.5,1946.3,2017.4,2088.5,2160.6,2232.1])
    BaseLogPartFctRef[ 287 ]['NbLabels']= 3
    BaseLogPartFctRef[ 287 ]['NbCliques']= 1464
    BaseLogPartFctRef[ 287 ]['NbSites']= 660
    BaseLogPartFctRef[ 287 ]['StdNgbhDivMoyNgbh']= 0.268932478729
    BaseLogPartFctRef[288 ]={}
    BaseLogPartFctRef[288]['LogPF']=np.array([695.4,719.2,743.1,768.0,793.4,820.1,848.3,876.3,906.9,940.0,975.5,1016.8,1063.1,1111.4,1164.1,1217.2,1276.1,1335.6,1396.4,1459.0,1522.7,1587.6,1651.9,1717.0,1783.6,1850.8,1918.5,1986.1,2053.6,2121.8])
    BaseLogPartFctRef[ 288 ]['NbLabels']= 3
    BaseLogPartFctRef[ 288 ]['NbCliques']= 1389
    BaseLogPartFctRef[ 288 ]['NbSites']= 633
    BaseLogPartFctRef[ 288 ]['StdNgbhDivMoyNgbh']= 0.276406948361
    BaseLogPartFctRef[289 ]={}
    BaseLogPartFctRef[289]['LogPF']=np.array([643.8,665.3,688.0,711.9,737.0,762.2,787.8,814.7,843.0,874.0,907.4,945.1,984.6,1032.2,1081.8,1133.2,1186.8,1243.0,1297.5,1357.5,1418.3,1479.2,1539.5,1600.7,1663.3,1725.9,1788.4,1851.3,1915.2,1979.0])
    BaseLogPartFctRef[ 289 ]['NbLabels']= 3
    BaseLogPartFctRef[ 289 ]['NbCliques']= 1301
    BaseLogPartFctRef[ 289 ]['NbSites']= 586
    BaseLogPartFctRef[ 289 ]['StdNgbhDivMoyNgbh']= 0.279615990334
    BaseLogPartFctRef[290 ]={}
    BaseLogPartFctRef[290]['LogPF']=np.array([718.5,742.9,767.6,793.0,819.4,847.2,874.9,904.7,938.0,969.5,1008.4,1050.2,1097.8,1144.1,1195.6,1250.4,1308.7,1369.6,1433.3,1495.9,1560.2,1624.7,1691.5,1757.6,1825.9,1892.4,1960.9,2029.5,2098.6,2167.3])
    BaseLogPartFctRef[ 290 ]['NbLabels']= 3
    BaseLogPartFctRef[ 290 ]['NbCliques']= 1415
    BaseLogPartFctRef[ 290 ]['NbSites']= 654
    BaseLogPartFctRef[ 290 ]['StdNgbhDivMoyNgbh']= 0.280427716308
    BaseLogPartFctRef[291 ]={}
    BaseLogPartFctRef[291]['LogPF']=np.array([705.3,729.4,753.6,778.9,804.8,832.3,860.4,889.2,918.8,950.8,987.0,1027.1,1073.6,1124.6,1176.8,1231.5,1290.1,1351.3,1412.8,1475.9,1538.5,1603.1,1669.9,1735.8,1802.1,1869.1,1936.2,2004.5,2072.6,2140.9])
    BaseLogPartFctRef[ 291 ]['NbLabels']= 3
    BaseLogPartFctRef[ 291 ]['NbCliques']= 1396
    BaseLogPartFctRef[ 291 ]['NbSites']= 642
    BaseLogPartFctRef[ 291 ]['StdNgbhDivMoyNgbh']= 0.274809819809
    BaseLogPartFctRef[292 ]={}
    BaseLogPartFctRef[292]['LogPF']=np.array([637.2,658.2,680.2,703.0,726.4,751.1,776.2,802.6,830.1,859.6,891.8,925.9,965.3,1007.4,1054.7,1105.5,1155.1,1211.0,1266.0,1323.3,1381.2,1439.5,1498.0,1557.5,1617.6,1678.1,1739.1,1800.3,1861.1,1922.2])
    BaseLogPartFctRef[ 292 ]['NbLabels']= 3
    BaseLogPartFctRef[ 292 ]['NbCliques']= 1262
    BaseLogPartFctRef[ 292 ]['NbSites']= 580
    BaseLogPartFctRef[ 292 ]['StdNgbhDivMoyNgbh']= 0.286430368058
    BaseLogPartFctRef[293 ]={}
    BaseLogPartFctRef[293]['LogPF']=np.array([695.4,719.2,744.0,769.0,795.0,822.2,850.3,878.7,909.8,941.2,980.0,1020.3,1068.5,1116.8,1170.0,1222.6,1282.8,1342.1,1403.7,1466.6,1532.0,1596.1,1659.9,1726.3,1793.2,1861.2,1928.1,1996.4,2064.9,2133.1])
    BaseLogPartFctRef[ 293 ]['NbLabels']= 3
    BaseLogPartFctRef[ 293 ]['NbCliques']= 1399
    BaseLogPartFctRef[ 293 ]['NbSites']= 633
    BaseLogPartFctRef[ 293 ]['StdNgbhDivMoyNgbh']= 0.270857570679
    BaseLogPartFctRef[294 ]={}
    BaseLogPartFctRef[294]['LogPF']=np.array([692.1,716.0,740.7,766.5,793.2,820.1,847.7,876.7,907.6,939.6,976.2,1015.5,1057.5,1110.7,1163.0,1220.8,1279.5,1339.6,1402.3,1466.0,1530.7,1596.2,1662.3,1728.9,1795.3,1862.7,1930.2,1998.1,2067.1,2135.7])
    BaseLogPartFctRef[ 294 ]['NbLabels']= 3
    BaseLogPartFctRef[ 294 ]['NbCliques']= 1404
    BaseLogPartFctRef[ 294 ]['NbSites']= 630
    BaseLogPartFctRef[ 294 ]['StdNgbhDivMoyNgbh']= 0.279144327234
    BaseLogPartFctRef[295 ]={}
    BaseLogPartFctRef[295]['LogPF']=np.array([798.7,826.6,855.5,885.4,916.9,949.1,982.2,1018.0,1054.9,1095.1,1142.6,1196.1,1252.8,1309.6,1371.3,1438.1,1509.6,1580.3,1654.1,1729.5,1806.4,1883.6,1963.0,2042.2,2122.5,2202.1,2282.2,2363.7,2445.2,2527.2])
    BaseLogPartFctRef[ 295 ]['NbLabels']= 3
    BaseLogPartFctRef[ 295 ]['NbCliques']= 1662
    BaseLogPartFctRef[ 295 ]['NbSites']= 727
    BaseLogPartFctRef[ 295 ]['StdNgbhDivMoyNgbh']= 0.2566648115
    BaseLogPartFctRef[296 ]={}
    BaseLogPartFctRef[296]['LogPF']=np.array([705.3,729.0,754.1,778.7,804.5,831.5,859.6,889.0,920.4,953.5,987.2,1029.2,1078.3,1131.1,1184.6,1238.4,1297.3,1356.8,1417.6,1479.6,1543.2,1608.1,1673.6,1739.0,1805.9,1871.9,1938.9,2006.7,2074.3,2142.3])
    BaseLogPartFctRef[ 296 ]['NbLabels']= 3
    BaseLogPartFctRef[ 296 ]['NbCliques']= 1392
    BaseLogPartFctRef[ 296 ]['NbSites']= 642
    BaseLogPartFctRef[ 296 ]['StdNgbhDivMoyNgbh']= 0.27311390062
    BaseLogPartFctRef[297 ]={}
    BaseLogPartFctRef[297]['LogPF']=np.array([635.0,655.5,677.2,699.1,721.6,745.1,769.6,795.2,821.9,849.5,882.6,918.9,959.2,1000.9,1047.3,1095.1,1143.5,1194.9,1247.7,1300.9,1356.3,1411.4,1468.1,1525.7,1583.8,1641.7,1700.5,1759.6,1819.4,1878.6])
    BaseLogPartFctRef[ 297 ]['NbLabels']= 3
    BaseLogPartFctRef[ 297 ]['NbCliques']= 1224
    BaseLogPartFctRef[ 297 ]['NbSites']= 578
    BaseLogPartFctRef[ 297 ]['StdNgbhDivMoyNgbh']= 0.282448665186
    BaseLogPartFctRef[298 ]={}
    BaseLogPartFctRef[298]['LogPF']=np.array([785.5,812.7,841.2,869.8,900.0,930.6,962.7,996.0,1031.0,1070.1,1109.6,1158.4,1211.2,1267.9,1329.5,1393.6,1460.4,1531.4,1600.3,1674.2,1748.4,1822.5,1898.3,1974.0,2051.2,2129.3,2207.3,2285.4,2364.5,2443.6])
    BaseLogPartFctRef[ 298 ]['NbLabels']= 3
    BaseLogPartFctRef[ 298 ]['NbCliques']= 1606
    BaseLogPartFctRef[ 298 ]['NbSites']= 715
    BaseLogPartFctRef[ 298 ]['StdNgbhDivMoyNgbh']= 0.248105993557
    BaseLogPartFctRef[299 ]={}
    BaseLogPartFctRef[299]['LogPF']=np.array([821.8,851.0,881.0,912.3,944.9,978.2,1011.8,1048.7,1087.3,1130.1,1179.6,1230.6,1292.4,1356.2,1425.2,1498.2,1571.9,1650.6,1730.7,1811.3,1892.3,1974.6,2056.5,2139.6,2223.2,2307.8,2392.5,2477.5,2562.8,2647.9])
    BaseLogPartFctRef[ 299 ]['NbLabels']= 3
    BaseLogPartFctRef[ 299 ]['NbCliques']= 1737
    BaseLogPartFctRef[ 299 ]['NbSites']= 748
    BaseLogPartFctRef[ 299 ]['StdNgbhDivMoyNgbh']= 0.247301841432
    BaseLogPartFctRef[300 ]={}
    BaseLogPartFctRef[300]['LogPF']=np.array([680.0,702.3,725.2,748.8,773.8,799.1,825.5,853.3,881.6,911.1,944.8,981.5,1024.2,1070.1,1116.8,1169.2,1223.4,1278.1,1335.7,1394.8,1455.3,1517.1,1578.5,1641.8,1705.5,1769.7,1833.2,1897.3,1961.5,2026.6])
    BaseLogPartFctRef[ 300 ]['NbLabels']= 3
    BaseLogPartFctRef[ 300 ]['NbCliques']= 1324
    BaseLogPartFctRef[ 300 ]['NbSites']= 619
    BaseLogPartFctRef[ 300 ]['StdNgbhDivMoyNgbh']= 0.279879197433
    BaseLogPartFctRef[301 ]={}
    BaseLogPartFctRef[301]['LogPF']=np.array([717.4,741.6,766.7,793.1,820.1,847.6,876.9,907.6,940.3,973.4,1008.9,1052.1,1099.0,1151.2,1206.4,1262.7,1323.0,1386.4,1451.9,1516.0,1582.6,1649.5,1718.2,1787.1,1857.5,1927.3,1997.9,2068.4,2139.1,2210.3])
    BaseLogPartFctRef[ 301 ]['NbLabels']= 3
    BaseLogPartFctRef[ 301 ]['NbCliques']= 1454
    BaseLogPartFctRef[ 301 ]['NbSites']= 653
    BaseLogPartFctRef[ 301 ]['StdNgbhDivMoyNgbh']= 0.273239438847
    BaseLogPartFctRef[302 ]={}
    BaseLogPartFctRef[302]['LogPF']=np.array([767.9,793.9,820.2,848.1,876.3,905.6,936.0,967.7,1000.3,1037.5,1078.4,1124.2,1169.1,1222.9,1281.6,1342.7,1405.4,1470.1,1536.5,1604.9,1674.2,1744.6,1814.9,1887.2,1960.5,2034.0,2107.1,2180.7,2255.0,2329.8])
    BaseLogPartFctRef[ 302 ]['NbLabels']= 3
    BaseLogPartFctRef[ 302 ]['NbCliques']= 1522
    BaseLogPartFctRef[ 302 ]['NbSites']= 699
    BaseLogPartFctRef[ 302 ]['StdNgbhDivMoyNgbh']= 0.26619523446
    BaseLogPartFctRef[303 ]={}
    BaseLogPartFctRef[303]['LogPF']=np.array([776.7,802.8,829.5,856.9,885.3,915.4,946.1,977.9,1011.9,1049.4,1089.0,1136.3,1185.0,1240.0,1294.5,1355.0,1419.1,1484.6,1553.0,1621.9,1692.8,1763.6,1836.2,1908.5,1982.6,2056.2,2131.3,2205.6,2280.9,2356.7])
    BaseLogPartFctRef[ 303 ]['NbLabels']= 3
    BaseLogPartFctRef[ 303 ]['NbCliques']= 1543
    BaseLogPartFctRef[ 303 ]['NbSites']= 707
    BaseLogPartFctRef[ 303 ]['StdNgbhDivMoyNgbh']= 0.267395399121
    BaseLogPartFctRef[304 ]={}
    BaseLogPartFctRef[304]['LogPF']=np.array([732.8,756.7,780.8,806.5,832.7,859.8,888.2,916.8,948.3,979.0,1012.2,1050.8,1092.5,1137.1,1187.5,1240.7,1296.8,1354.0,1411.7,1474.4,1536.6,1599.0,1663.5,1729.1,1794.2,1859.9,1926.7,1994.4,2061.7,2128.6])
    BaseLogPartFctRef[ 304 ]['NbLabels']= 3
    BaseLogPartFctRef[ 304 ]['NbCliques']= 1392
    BaseLogPartFctRef[ 304 ]['NbSites']= 667
    BaseLogPartFctRef[ 304 ]['StdNgbhDivMoyNgbh']= 0.288462266949
    BaseLogPartFctRef[305 ]={}
    BaseLogPartFctRef[305]['LogPF']=np.array([777.8,804.6,832.3,860.1,889.5,920.5,951.0,983.6,1017.3,1054.0,1092.3,1136.0,1186.0,1237.7,1295.7,1359.9,1427.7,1492.2,1560.4,1632.3,1703.8,1777.0,1851.3,1925.5,2001.1,2076.9,2153.1,2229.3,2305.9,2383.1])
    BaseLogPartFctRef[ 305 ]['NbLabels']= 3
    BaseLogPartFctRef[ 305 ]['NbCliques']= 1569
    BaseLogPartFctRef[ 305 ]['NbSites']= 708
    BaseLogPartFctRef[ 305 ]['StdNgbhDivMoyNgbh']= 0.248564683861
    BaseLogPartFctRef[306 ]={}
    BaseLogPartFctRef[306]['LogPF']=np.array([717.4,741.8,767.4,793.5,820.5,848.9,877.9,908.9,941.5,977.4,1020.3,1063.1,1114.2,1165.5,1224.0,1283.2,1341.8,1405.3,1469.5,1535.1,1600.5,1668.8,1738.2,1807.5,1878.3,1948.9,2019.4,2090.6,2162.3,2234.2])
    BaseLogPartFctRef[ 306 ]['NbLabels']= 3
    BaseLogPartFctRef[ 306 ]['NbCliques']= 1461
    BaseLogPartFctRef[ 306 ]['NbSites']= 653
    BaseLogPartFctRef[ 306 ]['StdNgbhDivMoyNgbh']= 0.270948215172
    BaseLogPartFctRef[307 ]={}
    BaseLogPartFctRef[307]['LogPF']=np.array([683.3,705.6,729.7,753.7,779.3,805.0,831.9,859.1,887.9,920.0,954.9,989.1,1030.9,1080.2,1128.5,1180.2,1236.2,1293.2,1350.4,1410.1,1470.1,1532.4,1594.8,1656.7,1719.0,1783.2,1847.7,1912.5,1977.1,2042.2])
    BaseLogPartFctRef[ 307 ]['NbLabels']= 3
    BaseLogPartFctRef[ 307 ]['NbCliques']= 1335
    BaseLogPartFctRef[ 307 ]['NbSites']= 622
    BaseLogPartFctRef[ 307 ]['StdNgbhDivMoyNgbh']= 0.276450145133
    BaseLogPartFctRef[308 ]={}
    BaseLogPartFctRef[308]['LogPF']=np.array([795.4,822.3,850.0,879.0,908.8,939.8,971.2,1004.6,1039.0,1076.4,1119.3,1165.3,1217.1,1272.0,1332.9,1395.5,1457.3,1526.3,1598.1,1673.1,1746.4,1821.0,1896.0,1972.1,2048.6,2125.5,2203.4,2280.9,2358.8,2436.9])
    BaseLogPartFctRef[ 308 ]['NbLabels']= 3
    BaseLogPartFctRef[ 308 ]['NbCliques']= 1596
    BaseLogPartFctRef[ 308 ]['NbSites']= 724
    BaseLogPartFctRef[ 308 ]['StdNgbhDivMoyNgbh']= 0.258224305712
    BaseLogPartFctRef[309 ]={}
    BaseLogPartFctRef[309]['LogPF']=np.array([630.6,652.1,674.3,697.5,720.8,745.3,770.7,797.4,824.6,854.6,886.3,917.6,959.0,1003.4,1050.4,1100.6,1153.0,1206.7,1262.5,1318.7,1377.1,1435.3,1495.1,1554.9,1616.1,1677.4,1738.2,1800.3,1862.1,1924.1])
    BaseLogPartFctRef[ 309 ]['NbLabels']= 3
    BaseLogPartFctRef[ 309 ]['NbCliques']= 1266
    BaseLogPartFctRef[ 309 ]['NbSites']= 574
    BaseLogPartFctRef[ 309 ]['StdNgbhDivMoyNgbh']= 0.269653103733
    BaseLogPartFctRef[310 ]={}
    BaseLogPartFctRef[310]['LogPF']=np.array([662.5,684.0,707.2,729.6,753.8,778.7,804.8,831.8,859.7,890.6,922.5,957.7,994.7,1040.6,1085.4,1133.4,1183.9,1238.8,1296.3,1353.7,1412.4,1471.2,1532.0,1591.9,1653.2,1714.8,1776.5,1838.3,1900.4,1963.1])
    BaseLogPartFctRef[ 310 ]['NbLabels']= 3
    BaseLogPartFctRef[ 310 ]['NbCliques']= 1285
    BaseLogPartFctRef[ 310 ]['NbSites']= 603
    BaseLogPartFctRef[ 310 ]['StdNgbhDivMoyNgbh']= 0.291932109486
    BaseLogPartFctRef[311 ]={}
    BaseLogPartFctRef[311]['LogPF']=np.array([560.3,578.7,597.1,616.5,636.8,657.4,678.4,700.5,724.9,750.9,776.2,803.5,836.3,871.2,910.5,952.4,995.3,1040.0,1086.9,1134.4,1182.1,1230.0,1280.0,1329.8,1380.5,1431.6,1483.0,1534.7,1586.4,1638.9])
    BaseLogPartFctRef[ 311 ]['NbLabels']= 3
    BaseLogPartFctRef[ 311 ]['NbCliques']= 1071
    BaseLogPartFctRef[ 311 ]['NbSites']= 510
    BaseLogPartFctRef[ 311 ]['StdNgbhDivMoyNgbh']= 0.289240565804
    BaseLogPartFctRef[312 ]={}
    BaseLogPartFctRef[312]['LogPF']=np.array([785.5,811.6,839.6,867.6,897.5,928.4,960.1,993.0,1028.3,1068.4,1109.9,1154.2,1201.7,1256.7,1318.9,1384.1,1450.8,1521.5,1593.2,1665.4,1738.5,1811.0,1885.7,1960.7,2037.2,2114.2,2191.1,2268.1,2346.0,2424.1])
    BaseLogPartFctRef[ 312 ]['NbLabels']= 3
    BaseLogPartFctRef[ 312 ]['NbCliques']= 1592
    BaseLogPartFctRef[ 312 ]['NbSites']= 715
    BaseLogPartFctRef[ 312 ]['StdNgbhDivMoyNgbh']= 0.260541184501
    BaseLogPartFctRef[313 ]={}
    BaseLogPartFctRef[313]['LogPF']=np.array([660.3,681.9,704.7,727.9,751.9,776.3,802.6,829.2,857.4,888.0,920.6,957.8,997.8,1039.2,1087.7,1138.6,1188.3,1245.0,1302.1,1357.6,1416.1,1473.3,1533.7,1593.5,1654.2,1715.4,1776.8,1838.2,1900.7,1962.7])
    BaseLogPartFctRef[ 313 ]['NbLabels']= 3
    BaseLogPartFctRef[ 313 ]['NbCliques']= 1281
    BaseLogPartFctRef[ 313 ]['NbSites']= 601
    BaseLogPartFctRef[ 313 ]['StdNgbhDivMoyNgbh']= 0.29667269453
    BaseLogPartFctRef[314 ]={}
    BaseLogPartFctRef[314]['LogPF']=np.array([691.0,714.6,739.7,765.2,792.1,819.1,847.7,877.1,907.2,939.6,979.4,1022.9,1067.5,1119.7,1177.0,1233.5,1292.5,1355.9,1418.8,1482.8,1548.4,1613.4,1679.4,1745.9,1813.0,1881.1,1949.5,2018.5,2087.6,2157.0])
    BaseLogPartFctRef[ 314 ]['NbLabels']= 3
    BaseLogPartFctRef[ 314 ]['NbCliques']= 1410
    BaseLogPartFctRef[ 314 ]['NbSites']= 629
    BaseLogPartFctRef[ 314 ]['StdNgbhDivMoyNgbh']= 0.269776205412

    #In [4]: ComputeBaseLogPartFctRef_NonReg(FirstIndex=(255+60),NbExtraIndex=30,BetaMax=1.45,DeltaBeta=0.05,NbLabels=3,NBX=10,NBY=10,NBZ=10,BetaGeneration=0.3)
    BaseLogPartFctRef[315 ]={}
    BaseLogPartFctRef[315]['LogPF']=np.array([366.9,377.2,388.1,398.9,410.4,422.2,434.2,446.5,460.3,474.5,489.2,503.9,520.8,538.1,557.1,576.7,599.0,621.6,643.0,667.7,694.0,719.3,746.0,772.9,800.3,828.1,856.0,884.2,912.6,941.5])
    BaseLogPartFctRef[ 315 ]['NbLabels']= 3
    BaseLogPartFctRef[ 315 ]['NbCliques']= 615
    BaseLogPartFctRef[ 315 ]['NbSites']= 334
    BaseLogPartFctRef[ 315 ]['StdNgbhDivMoyNgbh']= 0.36598110595
    BaseLogPartFctRef[316 ]={}
    BaseLogPartFctRef[316]['LogPF']=np.array([114.3,116.9,119.8,122.8,125.8,128.8,132.1,135.5,139.1,142.6,146.4,150.4,154.4,158.6,163.0,167.6,172.3,177.2,182.2,187.7,193.1,198.6,204.7,210.9,217.8,224.6,231.4,238.4,245.5,253.1])
    BaseLogPartFctRef[ 316 ]['NbLabels']= 3
    BaseLogPartFctRef[ 316 ]['NbCliques']= 164
    BaseLogPartFctRef[ 316 ]['NbSites']= 104
    BaseLogPartFctRef[ 316 ]['StdNgbhDivMoyNgbh']= 0.363742189418
    BaseLogPartFctRef[317 ]={}
    BaseLogPartFctRef[317]['LogPF']=np.array([80.20,82.09,84.00,86.03,88.18,90.30,92.55,94.90,97.13,99.55,102.1,104.7,107.5,110.3,113.3,116.3,119.6,123.1,126.6,130.2,134.0,138.1,141.8,146.1,150.5,154.9,159.2,163.5,168.2,173.0])
    BaseLogPartFctRef[ 317 ]['NbLabels']= 3
    BaseLogPartFctRef[ 317 ]['NbCliques']= 110
    BaseLogPartFctRef[ 317 ]['NbSites']= 73
    BaseLogPartFctRef[ 317 ]['StdNgbhDivMoyNgbh']= 0.39988429009
    BaseLogPartFctRef[318 ]={}
    BaseLogPartFctRef[318]['LogPF']=np.array([81.30,83.19,85.15,87.18,89.26,91.49,93.68,95.88,98.28,100.7,103.2,105.9,108.7,111.7,114.7,118.2,121.3,124.7,128.2,131.8,135.7,139.9,144.1,148.3,152.5,157.3,162.1,166.7,171.5,176.4])
    BaseLogPartFctRef[ 318 ]['NbLabels']= 3
    BaseLogPartFctRef[ 318 ]['NbCliques']= 111
    BaseLogPartFctRef[ 318 ]['NbSites']= 74
    BaseLogPartFctRef[ 318 ]['StdNgbhDivMoyNgbh']= 0.40813745446
    BaseLogPartFctRef[319 ]={}
    BaseLogPartFctRef[319]['LogPF']=np.array([147.2,151.0,154.9,158.9,163.1,167.5,171.8,176.3,181.1,186.1,191.2,196.8,202.6,208.6,214.6,221.3,228.2,235.8,243.3,251.4,259.8,268.4,277.1,286.4,295.4,304.8,314.3,324.0,334.0,344.0])
    BaseLogPartFctRef[ 319 ]['NbLabels']= 3
    BaseLogPartFctRef[ 319 ]['NbCliques']= 221
    BaseLogPartFctRef[ 319 ]['NbSites']= 134
    BaseLogPartFctRef[ 319 ]['StdNgbhDivMoyNgbh']= 0.38560906263
    BaseLogPartFctRef[320 ]={}
    BaseLogPartFctRef[320]['LogPF']=np.array([126.3,129.5,132.7,136.0,139.6,143.2,147.0,150.7,154.6,158.7,163.0,167.4,171.9,177.0,182.2,187.6,193.3,199.2,205.5,211.2,218.0,225.3,232.5,240.2,247.6,255.3,263.2,271.5,280.0,288.5])
    BaseLogPartFctRef[ 320 ]['NbLabels']= 3
    BaseLogPartFctRef[ 320 ]['NbCliques']= 187
    BaseLogPartFctRef[ 320 ]['NbSites']= 115
    BaseLogPartFctRef[ 320 ]['StdNgbhDivMoyNgbh']= 0.350072864142
    BaseLogPartFctRef[321 ]={}
    BaseLogPartFctRef[321]['LogPF']=np.array([104.4,107.1,110.0,113.1,116.1,119.2,122.4,125.7,129.1,132.8,136.6,140.4,144.9,149.1,153.4,158.0,162.7,168.0,173.0,179.2,185.3,191.9,198.2,205.1,212.3,219.4,226.8,234.0,241.6,249.3])
    BaseLogPartFctRef[ 321 ]['NbLabels']= 3
    BaseLogPartFctRef[ 321 ]['NbCliques']= 163
    BaseLogPartFctRef[ 321 ]['NbSites']= 95
    BaseLogPartFctRef[ 321 ]['StdNgbhDivMoyNgbh']= 0.382651352394
    BaseLogPartFctRef[322 ]={}
    BaseLogPartFctRef[322]['LogPF']=np.array([316.4,325.0,333.7,343.1,352.7,362.8,372.6,382.7,393.4,405.2,417.6,430.3,443.3,457.4,473.8,489.8,508.4,527.1,546.6,566.8,585.9,607.1,628.4,650.9,673.5,696.5,719.9,743.6,767.5,791.6])
    BaseLogPartFctRef[ 322 ]['NbLabels']= 3
    BaseLogPartFctRef[ 322 ]['NbCliques']= 515
    BaseLogPartFctRef[ 322 ]['NbSites']= 288
    BaseLogPartFctRef[ 322 ]['StdNgbhDivMoyNgbh']= 0.365049934797
    BaseLogPartFctRef[323 ]={}
    BaseLogPartFctRef[323]['LogPF']=np.array([207.6,213.5,219.6,225.8,232.2,238.9,245.9,253.2,260.8,268.4,276.5,285.0,294.3,303.8,313.4,323.7,335.3,346.9,358.8,371.5,385.4,399.8,414.2,429.6,444.7,460.4,476.3,492.3,508.4,524.7])
    BaseLogPartFctRef[ 323 ]['NbLabels']= 3
    BaseLogPartFctRef[ 323 ]['NbCliques']= 345
    BaseLogPartFctRef[ 323 ]['NbSites']= 189
    BaseLogPartFctRef[ 323 ]['StdNgbhDivMoyNgbh']= 0.331011355529
    BaseLogPartFctRef[324 ]={}
    BaseLogPartFctRef[324]['LogPF']=np.array([209.8,214.8,220.0,225.4,231.0,236.9,242.9,249.0,255.3,261.9,268.8,275.9,283.3,291.0,299.2,308.0,316.9,326.8,336.6,347.1,357.7,368.8,380.0,391.6,403.6,416.4,429.0,441.9,455.2,468.8])
    BaseLogPartFctRef[ 324 ]['NbLabels']= 3
    BaseLogPartFctRef[ 324 ]['NbCliques']= 301
    BaseLogPartFctRef[ 324 ]['NbSites']= 191
    BaseLogPartFctRef[ 324 ]['StdNgbhDivMoyNgbh']= 0.368659147247
    BaseLogPartFctRef[325 ]={}
    BaseLogPartFctRef[325]['LogPF']=np.array([141.7,145.5,149.4,153.4,157.4,161.7,166.2,170.9,175.4,180.3,185.4,190.6,196.1,201.7,208.1,214.7,221.5,228.4,236.0,243.8,252.0,260.6,269.4,278.8,288.1,297.5,307.2,317.1,327.0,337.1])
    BaseLogPartFctRef[ 325 ]['NbLabels']= 3
    BaseLogPartFctRef[ 325 ]['NbCliques']= 218
    BaseLogPartFctRef[ 325 ]['NbSites']= 129
    BaseLogPartFctRef[ 325 ]['StdNgbhDivMoyNgbh']= 0.387193491953
    BaseLogPartFctRef[326 ]={}
    BaseLogPartFctRef[326]['LogPF']=np.array([234.0,240.5,247.3,254.3,261.4,269.2,277.0,284.9,293.4,301.9,310.5,319.7,329.8,340.7,352.1,365.1,378.7,393.0,407.6,423.3,439.0,455.2,471.6,488.5,505.5,523.3,541.6,559.4,577.8,595.9])
    BaseLogPartFctRef[ 326 ]['NbLabels']= 3
    BaseLogPartFctRef[ 326 ]['NbCliques']= 389
    BaseLogPartFctRef[ 326 ]['NbSites']= 213
    BaseLogPartFctRef[ 326 ]['StdNgbhDivMoyNgbh']= 0.325694986108
    BaseLogPartFctRef[327 ]={}
    BaseLogPartFctRef[327]['LogPF']=np.array([294.4,302.2,310.4,318.8,327.3,336.3,345.4,355.0,365.2,375.8,386.5,397.3,409.4,422.6,437.2,451.3,466.8,481.4,498.5,515.5,534.0,553.3,572.9,592.7,613.4,634.3,654.9,675.4,696.5,718.1])
    BaseLogPartFctRef[ 327 ]['NbLabels']= 3
    BaseLogPartFctRef[ 327 ]['NbCliques']= 465
    BaseLogPartFctRef[ 327 ]['NbSites']= 268
    BaseLogPartFctRef[ 327 ]['StdNgbhDivMoyNgbh']= 0.365667449166
    BaseLogPartFctRef[328 ]={}
    BaseLogPartFctRef[328]['LogPF']=np.array([292.2,300.2,308.6,317.1,326.0,335.1,344.7,354.2,364.2,374.8,386.1,397.0,408.9,422.2,436.4,450.6,465.8,482.3,499.2,517.6,536.7,555.6,576.0,596.3,617.8,639.7,661.3,683.4,705.4,727.8])
    BaseLogPartFctRef[ 328 ]['NbLabels']= 3
    BaseLogPartFctRef[ 328 ]['NbCliques']= 476
    BaseLogPartFctRef[ 328 ]['NbSites']= 266
    BaseLogPartFctRef[ 328 ]['StdNgbhDivMoyNgbh']= 0.337418413588
    BaseLogPartFctRef[329 ]={}
    BaseLogPartFctRef[329]['LogPF']=np.array([107.7,110.5,113.4,116.4,119.4,122.7,126.0,129.4,133.0,136.5,140.5,144.9,148.9,153.2,157.6,162.5,167.1,172.0,177.4,182.9,188.9,195.2,201.6,208.3,215.1,222.4,229.4,236.9,244.4,251.8])
    BaseLogPartFctRef[ 329 ]['NbLabels']= 3
    BaseLogPartFctRef[ 329 ]['NbCliques']= 164
    BaseLogPartFctRef[ 329 ]['NbSites']= 98
    BaseLogPartFctRef[ 329 ]['StdNgbhDivMoyNgbh']= 0.363676245603
    BaseLogPartFctRef[330 ]={}
    BaseLogPartFctRef[330]['LogPF']=np.array([130.7,134.0,137.4,140.8,144.4,148.2,151.9,155.8,159.6,163.9,168.5,173.2,178.0,183.0,188.3,193.8,199.5,206.0,212.3,219.2,226.3,233.1,240.2,247.9,256.0,264.0,272.2,280.8,289.3,297.9])
    BaseLogPartFctRef[ 330 ]['NbLabels']= 3
    BaseLogPartFctRef[ 330 ]['NbCliques']= 192
    BaseLogPartFctRef[ 330 ]['NbSites']= 119
    BaseLogPartFctRef[ 330 ]['StdNgbhDivMoyNgbh']= 0.372511651842
    BaseLogPartFctRef[331 ]={}
    BaseLogPartFctRef[331]['LogPF']=np.array([336.2,346.0,356.4,366.7,377.6,388.6,400.1,412.2,424.8,437.5,451.2,466.2,481.2,496.8,515.9,535.2,555.4,576.3,598.7,621.3,644.9,670.5,695.7,722.1,748.8,775.6,803.3,830.4,857.7,885.8])
    BaseLogPartFctRef[ 331 ]['NbLabels']= 3
    BaseLogPartFctRef[ 331 ]['NbCliques']= 580
    BaseLogPartFctRef[ 331 ]['NbSites']= 306
    BaseLogPartFctRef[ 331 ]['StdNgbhDivMoyNgbh']= 0.329518283642
    BaseLogPartFctRef[332 ]={}
    BaseLogPartFctRef[332]['LogPF']=np.array([167.0,171.1,175.3,179.6,184.2,188.9,193.9,198.8,204.0,209.3,214.8,220.6,226.4,232.9,239.3,246.4,253.6,261.1,269.1,277.1,285.7,294.4,303.2,312.6,322.6,332.2,342.1,352.7,363.4,374.2])
    BaseLogPartFctRef[ 332 ]['NbLabels']= 3
    BaseLogPartFctRef[ 332 ]['NbCliques']= 242
    BaseLogPartFctRef[ 332 ]['NbSites']= 152
    BaseLogPartFctRef[ 332 ]['StdNgbhDivMoyNgbh']= 0.360403372577
    BaseLogPartFctRef[333 ]={}
    BaseLogPartFctRef[333]['LogPF']=np.array([218.6,223.9,229.4,235.2,241.2,247.3,253.5,260.2,267.0,274.3,281.6,289.2,296.9,305.0,313.8,323.0,332.1,342.0,352.2,363.0,374.5,385.9,398.4,410.6,423.3,436.7,449.9,463.6,477.6,491.5])
    BaseLogPartFctRef[ 333 ]['NbLabels']= 3
    BaseLogPartFctRef[ 333 ]['NbCliques']= 316
    BaseLogPartFctRef[ 333 ]['NbSites']= 199
    BaseLogPartFctRef[ 333 ]['StdNgbhDivMoyNgbh']= 0.391555179039
    BaseLogPartFctRef[334 ]={}
    BaseLogPartFctRef[334]['LogPF']=np.array([151.6,155.6,159.7,163.8,168.2,172.5,177.2,181.9,186.7,192.0,197.5,202.9,208.6,214.5,220.9,227.3,233.9,241.1,248.7,256.9,265.4,274.4,283.7,293.1,302.5,312.3,322.3,332.7,342.6,353.1])
    BaseLogPartFctRef[ 334 ]['NbLabels']= 3
    BaseLogPartFctRef[ 334 ]['NbCliques']= 229
    BaseLogPartFctRef[ 334 ]['NbSites']= 138
    BaseLogPartFctRef[ 334 ]['StdNgbhDivMoyNgbh']= 0.354636934759
    BaseLogPartFctRef[335 ]={}
    BaseLogPartFctRef[335]['LogPF']=np.array([99.97,102.7,105.4,108.3,111.4,114.4,117.5,120.9,124.4,127.9,131.5,135.4,139.5,143.5,147.7,152.0,157.2,162.3,168.0,173.8,179.5,185.6,191.7,198.1,204.9,211.9,218.9,226.0,233.4,240.7])
    BaseLogPartFctRef[ 335 ]['NbLabels']= 3
    BaseLogPartFctRef[ 335 ]['NbCliques']= 159
    BaseLogPartFctRef[ 335 ]['NbSites']= 91
    BaseLogPartFctRef[ 335 ]['StdNgbhDivMoyNgbh']= 0.33924306586
    BaseLogPartFctRef[336 ]={}
    BaseLogPartFctRef[336]['LogPF']=np.array([90.09,92.50,95.06,97.71,100.3,103.1,105.9,108.8,111.9,115.3,118.7,122.1,125.8,129.5,133.6,138.1,143.2,148.2,153.6,159.0,164.4,170.0,176.1,182.3,188.6,195.0,201.6,208.2,215.0,221.6])
    BaseLogPartFctRef[ 336 ]['NbLabels']= 3
    BaseLogPartFctRef[ 336 ]['NbCliques']= 144
    BaseLogPartFctRef[ 336 ]['NbSites']= 82
    BaseLogPartFctRef[ 336 ]['StdNgbhDivMoyNgbh']= 0.362854513411
    BaseLogPartFctRef[337 ]={}
    BaseLogPartFctRef[337]['LogPF']=np.array([60.42,61.87,63.44,65.10,66.80,68.56,70.46,72.47,74.42,76.39,78.39,80.60,82.91,85.38,87.92,90.45,93.06,96.24,99.38,102.4,105.6,108.6,112.4,116.2,120.0,124.1,127.9,131.8,135.9,140.1])
    BaseLogPartFctRef[ 337 ]['NbLabels']= 3
    BaseLogPartFctRef[ 337 ]['NbCliques']= 90
    BaseLogPartFctRef[ 337 ]['NbSites']= 55
    BaseLogPartFctRef[ 337 ]['StdNgbhDivMoyNgbh']= 0.38569460792
    BaseLogPartFctRef[338 ]={}
    BaseLogPartFctRef[338]['LogPF']=np.array([148.3,152.3,156.4,160.6,165.1,169.9,174.6,179.6,184.7,190.0,195.4,201.2,207.5,213.8,220.5,228.0,236.1,244.7,253.4,262.4,272.1,281.4,291.0,301.2,311.9,322.6,333.4,344.1,354.9,366.0])
    BaseLogPartFctRef[ 338 ]['NbLabels']= 3
    BaseLogPartFctRef[ 338 ]['NbCliques']= 239
    BaseLogPartFctRef[ 338 ]['NbSites']= 135
    BaseLogPartFctRef[ 338 ]['StdNgbhDivMoyNgbh']= 0.378058229611
    BaseLogPartFctRef[339 ]={}
    BaseLogPartFctRef[339]['LogPF']=np.array([81.30,83.23,85.26,87.31,89.42,91.62,93.94,96.20,98.54,101.0,103.7,106.6,109.4,112.3,115.7,118.9,122.2,126.2,130.0,133.8,137.8,141.9,146.2,150.6,155.1,159.6,164.5,169.4,174.2,179.3])
    BaseLogPartFctRef[ 339 ]['NbLabels']= 3
    BaseLogPartFctRef[ 339 ]['NbCliques']= 115
    BaseLogPartFctRef[ 339 ]['NbSites']= 74
    BaseLogPartFctRef[ 339 ]['StdNgbhDivMoyNgbh']= 0.367360929892
    BaseLogPartFctRef[340 ]={}
    BaseLogPartFctRef[340]['LogPF']=np.array([450.4,463.6,477.2,490.9,506.0,521.2,536.5,553.1,569.8,587.4,607.3,626.7,648.5,671.8,696.1,725.1,751.4,779.6,811.0,841.8,875.1,910.3,945.5,981.5,1016.7,1052.5,1089.4,1126.5,1163.5,1200.9])
    BaseLogPartFctRef[ 340 ]['NbLabels']= 3
    BaseLogPartFctRef[ 340 ]['NbCliques']= 785
    BaseLogPartFctRef[ 340 ]['NbSites']= 410
    BaseLogPartFctRef[ 340 ]['StdNgbhDivMoyNgbh']= 0.312518467356
    BaseLogPartFctRef[341 ]={}
    BaseLogPartFctRef[341]['LogPF']=np.array([232.9,238.6,245.0,251.7,258.4,265.5,272.8,280.4,288.2,296.5,304.7,313.2,322.4,331.9,342.2,353.8,364.3,376.2,390.3,403.4,417.1,432.4,446.9,462.4,477.5,493.5,510.3,526.5,543.3,560.2])
    BaseLogPartFctRef[ 341 ]['NbLabels']= 3
    BaseLogPartFctRef[ 341 ]['NbCliques']= 363
    BaseLogPartFctRef[ 341 ]['NbSites']= 212
    BaseLogPartFctRef[ 341 ]['StdNgbhDivMoyNgbh']= 0.332688757734
    BaseLogPartFctRef[342 ]={}
    BaseLogPartFctRef[342]['LogPF']=np.array([260.4,267.7,275.2,283.0,290.9,299.5,308.5,317.6,327.1,337.0,347.4,358.4,371.1,384.1,398.4,412.2,426.8,441.6,458.6,475.6,494.2,513.5,532.0,550.8,570.7,590.9,610.7,631.1,652.0,672.6])
    BaseLogPartFctRef[ 342 ]['NbLabels']= 3
    BaseLogPartFctRef[ 342 ]['NbCliques']= 441
    BaseLogPartFctRef[ 342 ]['NbSites']= 237
    BaseLogPartFctRef[ 342 ]['StdNgbhDivMoyNgbh']= 0.386940124834
    BaseLogPartFctRef[343 ]={}
    BaseLogPartFctRef[343]['LogPF']=np.array([236.2,242.1,248.4,254.9,261.7,268.9,276.3,283.8,291.4,299.5,308.3,317.7,327.4,337.2,347.3,357.6,369.6,381.1,394.4,406.4,420.4,435.3,450.0,465.2,480.3,496.3,512.2,529.1,545.7,562.9])
    BaseLogPartFctRef[ 343 ]['NbLabels']= 3
    BaseLogPartFctRef[ 343 ]['NbCliques']= 365
    BaseLogPartFctRef[ 343 ]['NbSites']= 215
    BaseLogPartFctRef[ 343 ]['StdNgbhDivMoyNgbh']= 0.357302435905
    BaseLogPartFctRef[344 ]={}
    BaseLogPartFctRef[344]['LogPF']=np.array([107.7,110.4,113.1,115.9,118.9,121.9,125.1,128.1,131.5,135.1,138.8,142.5,146.4,150.7,154.9,159.2,164.3,169.3,174.4,180.0,185.6,191.6,197.9,204.3,210.7,217.8,225.0,231.8,238.9,246.2])
    BaseLogPartFctRef[ 344 ]['NbLabels']= 3
    BaseLogPartFctRef[ 344 ]['NbCliques']= 159
    BaseLogPartFctRef[ 344 ]['NbSites']= 98
    BaseLogPartFctRef[ 344 ]['StdNgbhDivMoyNgbh']= 0.385003124456

    #In [5]: ComputeBaseLogPartFctRef_NonReg(FirstIndex=(255+90),NbExtraIndex=30,BetaMax=1.45,DeltaBeta=0.05,NbLabels=3,NBX=10,NBY=10,NBZ=10,BetaGeneration=0.6)
    BaseLogPartFctRef[345 ]={}
    BaseLogPartFctRef[345]['LogPF']=np.array([1004.1,1043.5,1084.2,1126.3,1171.2,1215.7,1263.0,1314.5,1368.5,1427.8,1494.6,1571.3,1659.2,1759.1,1857.7,1962.2,2068.9,2176.0,2285.6,2397.1,2509.4,2622.9,2737.1,2851.6,2966.1,3081.3,3196.6,3312.4,3428.2,3544.1])
    BaseLogPartFctRef[ 345 ]['NbLabels']= 3
    BaseLogPartFctRef[ 345 ]['NbCliques']= 2334
    BaseLogPartFctRef[ 345 ]['NbSites']= 914
    BaseLogPartFctRef[ 345 ]['StdNgbhDivMoyNgbh']= 0.184520051268
    BaseLogPartFctRef[346 ]={}
    BaseLogPartFctRef[346]['LogPF']=np.array([1018.4,1058.8,1100.6,1143.0,1187.3,1233.0,1280.8,1332.5,1388.4,1453.6,1523.1,1605.0,1687.7,1781.5,1884.0,1987.9,2095.0,2204.2,2316.2,2428.8,2541.4,2655.9,2771.3,2887.1,3004.2,3120.9,3237.6,3354.9,3472.2,3589.8])
    BaseLogPartFctRef[ 346 ]['NbLabels']= 3
    BaseLogPartFctRef[ 346 ]['NbCliques']= 2366
    BaseLogPartFctRef[ 346 ]['NbSites']= 927
    BaseLogPartFctRef[ 346 ]['StdNgbhDivMoyNgbh']= 0.177286604214
    BaseLogPartFctRef[347 ]={}
    BaseLogPartFctRef[347]['LogPF']=np.array([1008.5,1048.1,1088.7,1131.6,1176.0,1220.9,1267.1,1319.5,1376.9,1439.5,1511.0,1587.1,1672.6,1765.5,1865.3,1965.9,2072.0,2182.1,2293.7,2404.4,2516.8,2629.5,2742.6,2857.0,2971.9,3087.2,3203.1,3319.3,3435.2,3551.8])
    BaseLogPartFctRef[ 347 ]['NbLabels']= 3
    BaseLogPartFctRef[ 347 ]['NbCliques']= 2337
    BaseLogPartFctRef[ 347 ]['NbSites']= 918
    BaseLogPartFctRef[ 347 ]['StdNgbhDivMoyNgbh']= 0.182266929196
    BaseLogPartFctRef[348 ]={}
    BaseLogPartFctRef[348]['LogPF']=np.array([1030.5,1070.9,1113.3,1157.0,1201.8,1248.9,1297.8,1348.1,1404.8,1466.7,1541.0,1630.4,1724.4,1819.8,1924.2,2031.6,2140.2,2253.0,2367.4,2483.5,2600.4,2717.9,2837.0,2955.8,3075.2,3194.0,3314.3,3435.2,3556.1,3676.8])
    BaseLogPartFctRef[ 348 ]['NbLabels']= 3
    BaseLogPartFctRef[ 348 ]['NbCliques']= 2424
    BaseLogPartFctRef[ 348 ]['NbSites']= 938
    BaseLogPartFctRef[ 348 ]['StdNgbhDivMoyNgbh']= 0.173444289427
    BaseLogPartFctRef[349 ]={}
    BaseLogPartFctRef[349]['LogPF']=np.array([1027.2,1067.7,1109.3,1152.4,1196.8,1242.8,1290.4,1342.5,1396.1,1456.1,1535.3,1614.4,1706.5,1800.5,1905.4,2010.0,2119.2,2232.0,2345.8,2460.5,2575.3,2692.3,2810.1,2928.8,3047.8,3166.5,3285.7,3405.2,3524.6,3644.6])
    BaseLogPartFctRef[ 349 ]['NbLabels']= 3
    BaseLogPartFctRef[ 349 ]['NbCliques']= 2409
    BaseLogPartFctRef[ 349 ]['NbSites']= 935
    BaseLogPartFctRef[ 349 ]['StdNgbhDivMoyNgbh']= 0.173752738849
    BaseLogPartFctRef[350 ]={}
    BaseLogPartFctRef[350]['LogPF']=np.array([1022.8,1063.4,1106.2,1148.6,1193.7,1240.6,1288.8,1340.4,1393.8,1453.9,1530.0,1615.0,1705.3,1806.4,1908.8,2018.0,2125.8,2238.3,2351.2,2465.9,2580.9,2697.8,2815.6,2932.9,3051.1,3169.4,3288.2,3407.4,3526.4,3645.8])
    BaseLogPartFctRef[ 350 ]['NbLabels']= 3
    BaseLogPartFctRef[ 350 ]['NbCliques']= 2396
    BaseLogPartFctRef[ 350 ]['NbSites']= 931
    BaseLogPartFctRef[ 350 ]['StdNgbhDivMoyNgbh']= 0.16981940557
    BaseLogPartFctRef[351 ]={}
    BaseLogPartFctRef[351]['LogPF']=np.array([1021.7,1061.8,1103.3,1147.2,1190.6,1236.9,1284.7,1335.0,1390.6,1451.2,1524.1,1607.1,1699.5,1798.5,1901.5,2006.8,2117.9,2228.8,2338.2,2451.9,2565.9,2681.6,2797.6,2915.2,3031.5,3149.0,3266.8,3384.5,3503.0,3621.3])
    BaseLogPartFctRef[ 351 ]['NbLabels']= 3
    BaseLogPartFctRef[ 351 ]['NbCliques']= 2384
    BaseLogPartFctRef[ 351 ]['NbSites']= 930
    BaseLogPartFctRef[ 351 ]['StdNgbhDivMoyNgbh']= 0.178554698558
    BaseLogPartFctRef[352 ]={}
    BaseLogPartFctRef[352]['LogPF']=np.array([1049.2,1091.5,1134.9,1179.9,1225.7,1273.2,1323.6,1376.8,1435.2,1497.8,1577.9,1669.8,1763.4,1861.3,1966.6,2077.1,2192.3,2309.9,2426.2,2543.6,2663.2,2784.4,2906.2,3028.3,3151.2,3274.3,3397.4,3520.4,3644.0,3768.0])
    BaseLogPartFctRef[ 352 ]['NbLabels']= 3
    BaseLogPartFctRef[ 352 ]['NbCliques']= 2487
    BaseLogPartFctRef[ 352 ]['NbSites']= 955
    BaseLogPartFctRef[ 352 ]['StdNgbhDivMoyNgbh']= 0.162737513894
    BaseLogPartFctRef[353 ]={}
    BaseLogPartFctRef[353]['LogPF']=np.array([1021.7,1062.4,1105.4,1148.4,1193.4,1239.0,1287.2,1337.1,1390.0,1447.7,1519.0,1604.4,1693.9,1788.3,1893.0,2002.7,2112.3,2223.2,2336.6,2452.1,2566.7,2682.0,2798.1,2915.4,3033.0,3150.7,3269.5,3388.2,3507.1,3625.8])
    BaseLogPartFctRef[ 353 ]['NbLabels']= 3
    BaseLogPartFctRef[ 353 ]['NbCliques']= 2393
    BaseLogPartFctRef[ 353 ]['NbSites']= 930
    BaseLogPartFctRef[ 353 ]['StdNgbhDivMoyNgbh']= 0.171279097606
    BaseLogPartFctRef[354 ]={}
    BaseLogPartFctRef[354]['LogPF']=np.array([1043.7,1085.0,1127.2,1171.5,1217.8,1265.0,1313.9,1365.6,1421.8,1487.5,1563.7,1648.9,1741.8,1838.0,1945.5,2053.4,2166.7,2279.3,2395.2,2511.9,2631.0,2750.0,2869.0,2989.3,3110.5,3231.9,3353.8,3475.5,3597.8,3720.1])
    BaseLogPartFctRef[ 354 ]['NbLabels']= 3
    BaseLogPartFctRef[ 354 ]['NbCliques']= 2458
    BaseLogPartFctRef[ 354 ]['NbSites']= 950
    BaseLogPartFctRef[ 354 ]['StdNgbhDivMoyNgbh']= 0.16747255202
    BaseLogPartFctRef[355 ]={}
    BaseLogPartFctRef[355]['LogPF']=np.array([1020.6,1060.3,1102.0,1145.4,1190.1,1235.7,1284.4,1334.7,1391.5,1450.2,1527.6,1607.0,1702.3,1800.2,1903.8,2008.4,2118.9,2229.3,2342.7,2457.0,2571.6,2688.0,2804.9,2921.9,3040.0,3158.3,3276.3,3394.6,3513.5,3632.5])
    BaseLogPartFctRef[ 355 ]['NbLabels']= 3
    BaseLogPartFctRef[ 355 ]['NbCliques']= 2391
    BaseLogPartFctRef[ 355 ]['NbSites']= 929
    BaseLogPartFctRef[ 355 ]['StdNgbhDivMoyNgbh']= 0.177453346132
    BaseLogPartFctRef[356 ]={}
    BaseLogPartFctRef[356]['LogPF']=np.array([1037.1,1078.0,1121.0,1165.6,1211.4,1258.6,1308.4,1359.7,1415.6,1478.1,1555.0,1639.7,1736.0,1836.0,1942.7,2053.9,2169.2,2285.1,2402.9,2519.1,2637.8,2758.7,2877.9,2998.6,3119.3,3240.6,3361.9,3483.4,3605.5,3727.7])
    BaseLogPartFctRef[ 356 ]['NbLabels']= 3
    BaseLogPartFctRef[ 356 ]['NbCliques']= 2455
    BaseLogPartFctRef[ 356 ]['NbSites']= 944
    BaseLogPartFctRef[ 356 ]['StdNgbhDivMoyNgbh']= 0.167934586556
    BaseLogPartFctRef[357 ]={}
    BaseLogPartFctRef[357]['LogPF']=np.array([1044.8,1087.0,1130.1,1174.7,1221.5,1268.5,1318.0,1369.8,1427.7,1494.3,1565.5,1657.4,1755.9,1858.6,1969.6,2077.3,2187.7,2303.5,2420.9,2540.2,2657.8,2779.0,2899.9,3021.3,3143.2,3264.7,3387.0,3509.8,3632.6,3755.5])
    BaseLogPartFctRef[ 357 ]['NbLabels']= 3
    BaseLogPartFctRef[ 357 ]['NbCliques']= 2474
    BaseLogPartFctRef[ 357 ]['NbSites']= 951
    BaseLogPartFctRef[ 357 ]['StdNgbhDivMoyNgbh']= 0.159926437281
    BaseLogPartFctRef[358 ]={}
    BaseLogPartFctRef[358]['LogPF']=np.array([1027.2,1068.7,1110.8,1154.3,1199.3,1246.3,1294.3,1345.3,1398.9,1463.1,1530.8,1614.8,1707.3,1805.0,1907.5,2012.8,2122.8,2235.2,2348.0,2460.8,2575.3,2690.8,2807.7,2925.6,3043.4,3161.9,3281.1,3399.8,3519.0,3638.4])
    BaseLogPartFctRef[ 358 ]['NbLabels']= 3
    BaseLogPartFctRef[ 358 ]['NbCliques']= 2398
    BaseLogPartFctRef[ 358 ]['NbSites']= 935
    BaseLogPartFctRef[ 358 ]['StdNgbhDivMoyNgbh']= 0.168424957433
    BaseLogPartFctRef[359 ]={}
    BaseLogPartFctRef[359]['LogPF']=np.array([1005.2,1044.5,1084.6,1126.3,1169.2,1214.1,1261.3,1310.8,1367.8,1429.1,1494.6,1579.2,1660.0,1752.5,1853.8,1953.7,2062.1,2171.8,2281.7,2394.6,2506.9,2620.5,2734.9,2848.5,2963.6,3078.8,3194.5,3309.9,3426.2,3542.5])
    BaseLogPartFctRef[ 359 ]['NbLabels']= 3
    BaseLogPartFctRef[ 359 ]['NbCliques']= 2335
    BaseLogPartFctRef[ 359 ]['NbSites']= 915
    BaseLogPartFctRef[ 359 ]['StdNgbhDivMoyNgbh']= 0.185181218268
    BaseLogPartFctRef[360 ]={}
    BaseLogPartFctRef[360]['LogPF']=np.array([999.7,1039.3,1078.8,1119.7,1163.3,1208.3,1254.9,1302.4,1352.9,1416.3,1486.6,1566.9,1656.7,1753.3,1852.5,1955.1,2059.3,2166.0,2274.1,2384.5,2496.6,2608.4,2721.7,2835.1,2949.3,3063.8,3178.6,3293.4,3408.5,3524.1])
    BaseLogPartFctRef[ 360 ]['NbLabels']= 3
    BaseLogPartFctRef[ 360 ]['NbCliques']= 2318
    BaseLogPartFctRef[ 360 ]['NbSites']= 910
    BaseLogPartFctRef[ 360 ]['StdNgbhDivMoyNgbh']= 0.178226331535
    BaseLogPartFctRef[361 ]={}
    BaseLogPartFctRef[361]['LogPF']=np.array([1044.8,1087.3,1130.7,1174.9,1221.0,1270.3,1321.3,1375.1,1434.9,1498.8,1575.2,1663.4,1757.9,1861.7,1970.0,2081.3,2198.2,2313.7,2432.1,2551.0,2672.0,2793.0,2915.0,3037.7,3160.5,3283.9,3408.2,3531.9,3656.0,3779.9])
    BaseLogPartFctRef[ 361 ]['NbLabels']= 3
    BaseLogPartFctRef[ 361 ]['NbCliques']= 2492
    BaseLogPartFctRef[ 361 ]['NbSites']= 951
    BaseLogPartFctRef[ 361 ]['StdNgbhDivMoyNgbh']= 0.154629683739
    BaseLogPartFctRef[362 ]={}
    BaseLogPartFctRef[362]['LogPF']=np.array([1004.1,1044.2,1085.4,1128.0,1172.5,1217.5,1264.9,1314.3,1366.5,1429.4,1504.5,1584.0,1675.4,1770.6,1869.5,1974.5,2081.1,2188.0,2298.1,2408.1,2521.4,2636.5,2750.5,2865.0,2980.4,3096.1,3211.6,3328.1,3444.3,3560.7])
    BaseLogPartFctRef[ 362 ]['NbLabels']= 3
    BaseLogPartFctRef[ 362 ]['NbCliques']= 2339
    BaseLogPartFctRef[ 362 ]['NbSites']= 914
    BaseLogPartFctRef[ 362 ]['StdNgbhDivMoyNgbh']= 0.181130100505
    BaseLogPartFctRef[363 ]={}
    BaseLogPartFctRef[363]['LogPF']=np.array([1009.6,1049.8,1091.4,1134.6,1178.4,1223.9,1271.1,1321.1,1374.6,1433.9,1502.3,1590.1,1682.6,1775.7,1875.2,1981.6,2088.9,2197.8,2310.8,2423.5,2537.7,2653.5,2768.6,2885.0,3001.2,3118.2,3236.1,3353.0,3470.8,3589.1])
    BaseLogPartFctRef[ 363 ]['NbLabels']= 3
    BaseLogPartFctRef[ 363 ]['NbCliques']= 2376
    BaseLogPartFctRef[ 363 ]['NbSites']= 919
    BaseLogPartFctRef[ 363 ]['StdNgbhDivMoyNgbh']= 0.180711050403
    BaseLogPartFctRef[364 ]={}
    BaseLogPartFctRef[364]['LogPF']=np.array([1014.0,1053.8,1095.0,1138.0,1182.2,1228.5,1275.0,1325.0,1378.9,1439.9,1512.2,1599.5,1691.3,1790.0,1891.0,1997.1,2108.1,2218.1,2328.8,2443.4,2558.0,2673.3,2789.9,2905.9,3022.7,3139.8,3257.0,3374.3,3491.9,3609.7])
    BaseLogPartFctRef[ 364 ]['NbLabels']= 3
    BaseLogPartFctRef[ 364 ]['NbCliques']= 2374
    BaseLogPartFctRef[ 364 ]['NbSites']= 923
    BaseLogPartFctRef[ 364 ]['StdNgbhDivMoyNgbh']= 0.180087914656
    BaseLogPartFctRef[365 ]={}
    BaseLogPartFctRef[365]['LogPF']=np.array([1012.9,1052.3,1093.5,1135.2,1179.2,1225.4,1273.5,1322.0,1373.3,1438.5,1509.5,1584.3,1672.8,1768.1,1867.5,1972.7,2081.4,2193.3,2303.3,2414.5,2525.7,2640.6,2755.1,2871.2,2987.2,3103.1,3219.7,3336.3,3453.1,3570.2])
    BaseLogPartFctRef[ 365 ]['NbLabels']= 3
    BaseLogPartFctRef[ 365 ]['NbCliques']= 2352
    BaseLogPartFctRef[ 365 ]['NbSites']= 922
    BaseLogPartFctRef[ 365 ]['StdNgbhDivMoyNgbh']= 0.17804221797
    BaseLogPartFctRef[366 ]={}
    BaseLogPartFctRef[366]['LogPF']=np.array([1027.2,1068.6,1110.9,1154.2,1198.9,1246.0,1294.5,1344.1,1401.2,1461.7,1535.7,1617.1,1710.3,1809.3,1915.5,2024.9,2134.3,2247.4,2361.0,2475.1,2591.2,2709.6,2827.3,2945.2,3063.6,3182.9,3302.5,3422.2,3542.0,3662.0])
    BaseLogPartFctRef[ 366 ]['NbLabels']= 3
    BaseLogPartFctRef[ 366 ]['NbCliques']= 2413
    BaseLogPartFctRef[ 366 ]['NbSites']= 935
    BaseLogPartFctRef[ 366 ]['StdNgbhDivMoyNgbh']= 0.168958832152
    BaseLogPartFctRef[367 ]={}
    BaseLogPartFctRef[367]['LogPF']=np.array([1036.0,1077.8,1120.3,1164.2,1209.2,1256.6,1305.3,1356.4,1412.5,1475.1,1552.1,1631.2,1722.6,1822.1,1927.6,2035.9,2149.2,2262.2,2377.2,2494.6,2612.6,2730.9,2849.3,2967.6,3087.9,3208.2,3328.8,3449.7,3570.7,3692.0])
    BaseLogPartFctRef[ 367 ]['NbLabels']= 3
    BaseLogPartFctRef[ 367 ]['NbCliques']= 2435
    BaseLogPartFctRef[ 367 ]['NbSites']= 943
    BaseLogPartFctRef[ 367 ]['StdNgbhDivMoyNgbh']= 0.158793113095
    BaseLogPartFctRef[368 ]={}
    BaseLogPartFctRef[368]['LogPF']=np.array([998.6,1037.1,1077.1,1118.4,1161.1,1204.5,1249.7,1297.4,1349.4,1406.2,1470.2,1548.3,1634.2,1725.1,1818.8,1921.9,2021.0,2123.5,2229.0,2337.9,2448.8,2559.5,2670.2,2782.3,2894.4,3006.7,3119.6,3233.2,3346.4,3460.0])
    BaseLogPartFctRef[ 368 ]['NbLabels']= 3
    BaseLogPartFctRef[ 368 ]['NbCliques']= 2284
    BaseLogPartFctRef[ 368 ]['NbSites']= 909
    BaseLogPartFctRef[ 368 ]['StdNgbhDivMoyNgbh']= 0.184650972235
    BaseLogPartFctRef[369 ]={}
    BaseLogPartFctRef[369]['LogPF']=np.array([1010.7,1050.8,1091.9,1134.4,1178.6,1224.2,1272.4,1321.7,1374.5,1437.9,1508.8,1588.6,1678.4,1777.1,1876.9,1980.7,2088.4,2197.7,2307.4,2418.2,2530.8,2645.3,2759.8,2874.8,2991.0,3106.5,3222.6,3339.4,3455.9,3572.2])
    BaseLogPartFctRef[ 369 ]['NbLabels']= 3
    BaseLogPartFctRef[ 369 ]['NbCliques']= 2350
    BaseLogPartFctRef[ 369 ]['NbSites']= 920
    BaseLogPartFctRef[ 369 ]['StdNgbhDivMoyNgbh']= 0.174309545746
    BaseLogPartFctRef[370 ]={}
    BaseLogPartFctRef[370]['LogPF']=np.array([1016.2,1055.7,1097.0,1140.2,1184.0,1229.6,1278.0,1327.4,1381.6,1443.0,1512.4,1594.6,1685.4,1782.4,1886.0,1991.3,2096.8,2207.3,2319.9,2431.5,2545.3,2659.3,2774.2,2890.0,3006.1,3122.2,3238.5,3355.4,3472.5,3589.9])
    BaseLogPartFctRef[ 370 ]['NbLabels']= 3
    BaseLogPartFctRef[ 370 ]['NbCliques']= 2360
    BaseLogPartFctRef[ 370 ]['NbSites']= 925
    BaseLogPartFctRef[ 370 ]['StdNgbhDivMoyNgbh']= 0.177824088142
    BaseLogPartFctRef[371 ]={}
    BaseLogPartFctRef[371]['LogPF']=np.array([1036.0,1076.9,1119.4,1163.3,1209.1,1256.1,1305.3,1355.9,1412.8,1477.0,1551.5,1633.0,1724.9,1823.6,1927.3,2037.1,2146.0,2258.9,2373.7,2489.9,2606.5,2723.6,2842.0,2960.9,3080.3,3199.9,3320.0,3440.1,3560.0,3680.2])
    BaseLogPartFctRef[ 371 ]['NbLabels']= 3
    BaseLogPartFctRef[ 371 ]['NbCliques']= 2420
    BaseLogPartFctRef[ 371 ]['NbSites']= 943
    BaseLogPartFctRef[ 371 ]['StdNgbhDivMoyNgbh']= 0.163616515935
    BaseLogPartFctRef[372 ]={}
    BaseLogPartFctRef[372]['LogPF']=np.array([1017.3,1056.9,1098.5,1141.1,1184.6,1231.2,1279.4,1330.7,1383.6,1441.7,1512.0,1595.1,1683.0,1779.5,1881.6,1986.2,2095.9,2208.2,2318.7,2431.9,2545.7,2659.9,2775.8,2892.3,3009.2,3126.3,3243.6,3361.1,3478.7,3596.3])
    BaseLogPartFctRef[ 372 ]['NbLabels']= 3
    BaseLogPartFctRef[ 372 ]['NbCliques']= 2369
    BaseLogPartFctRef[ 372 ]['NbSites']= 926
    BaseLogPartFctRef[ 372 ]['StdNgbhDivMoyNgbh']= 0.169912475535
    BaseLogPartFctRef[373 ]={}
    BaseLogPartFctRef[373]['LogPF']=np.array([1015.1,1055.1,1095.8,1138.5,1183.0,1229.0,1275.3,1324.3,1375.4,1435.9,1511.5,1587.4,1679.3,1775.9,1875.5,1981.3,2087.8,2198.5,2308.9,2421.6,2534.0,2648.0,2762.4,2878.7,2994.5,3110.9,3227.6,3344.7,3462.3,3579.2])
    BaseLogPartFctRef[ 373 ]['NbLabels']= 3
    BaseLogPartFctRef[ 373 ]['NbCliques']= 2361
    BaseLogPartFctRef[ 373 ]['NbSites']= 924
    BaseLogPartFctRef[ 373 ]['StdNgbhDivMoyNgbh']= 0.170465756475
    BaseLogPartFctRef[374 ]={}
    BaseLogPartFctRef[374]['LogPF']=np.array([1017.3,1057.8,1099.8,1142.6,1186.0,1232.0,1280.3,1330.6,1385.5,1447.7,1525.9,1606.6,1694.2,1787.5,1888.3,1995.2,2100.7,2211.0,2322.1,2435.0,2548.7,2662.5,2777.4,2893.5,3010.1,3126.9,3243.7,3361.0,3478.7,3596.3])
    BaseLogPartFctRef[ 374 ]['NbLabels']= 3
    BaseLogPartFctRef[ 374 ]['NbCliques']= 2365
    BaseLogPartFctRef[ 374 ]['NbSites']= 926
    BaseLogPartFctRef[ 374 ]['StdNgbhDivMoyNgbh']= 0.176840058158

    #In [6]: ComputeBaseLogPartFctRef_NonReg(FirstIndex=(255+120),NbExtraIndex=30,BetaMax=1.45,DeltaBeta=0.05,NbLabels=3,NBX=15,NBY=15,NBZ=15,BetaGeneration=0.2)
    BaseLogPartFctRef[ 375 ]={}
    BaseLogPartFctRef[375]['LogPF']=np.array([50.54,51.58,52.63,53.71,54.85,56.04,57.22,58.50,59.78,61.13,62.51,63.94,65.41,66.99,68.57,70.17,71.92,73.66,75.43,77.24,79.19,81.18,83.15,85.28,87.42,89.62,91.80,94.13,96.51,98.91])
    BaseLogPartFctRef[ 375 ]['NbLabels']= 3
    BaseLogPartFctRef[ 375 ]['NbCliques']= 60
    BaseLogPartFctRef[ 375 ]['NbSites']= 46
    BaseLogPartFctRef[ 375 ]['StdNgbhDivMoyNgbh']= 0.43870513197
    BaseLogPartFctRef[376 ]={}
    BaseLogPartFctRef[376]['LogPF']=np.array([29.66,30.30,30.98,31.67,32.39,33.12,33.91,34.67,35.48,36.31,37.15,38.05,38.98,39.96,40.95,41.99,43.08,44.31,45.50,46.72,48.03,49.35,50.72,52.21,53.69,55.27,56.84,58.46,60.05,61.74])
    BaseLogPartFctRef[ 376 ]['NbLabels']= 3
    BaseLogPartFctRef[ 376 ]['NbCliques']= 38
    BaseLogPartFctRef[ 376 ]['NbSites']= 27
    BaseLogPartFctRef[ 376 ]['StdNgbhDivMoyNgbh']= 0.475103961735
    BaseLogPartFctRef[377 ]={}
    BaseLogPartFctRef[377]['LogPF']=np.array([32.96,33.53,34.14,34.76,35.42,36.10,36.81,37.52,38.25,39.04,39.80,40.64,41.46,42.32,43.19,44.11,45.07,46.04,47.02,48.02,49.10,50.21,51.29,52.50,53.65,54.85,56.08,57.41,58.69,59.98])
    BaseLogPartFctRef[ 377 ]['NbLabels']= 3
    BaseLogPartFctRef[ 377 ]['NbCliques']= 35
    BaseLogPartFctRef[ 377 ]['NbSites']= 30
    BaseLogPartFctRef[ 377 ]['StdNgbhDivMoyNgbh']= 0.37565966167
    BaseLogPartFctRef[378 ]={}
    BaseLogPartFctRef[378]['LogPF']=np.array([98.88,101.2,103.6,106.1,108.5,111.2,113.9,116.9,119.7,122.8,126.1,129.2,132.8,136.3,139.8,143.7,147.7,151.9,156.3,160.6,165.2,170.1,175.2,180.7,186.0,191.4,197.1,202.7,208.6,214.4])
    BaseLogPartFctRef[ 378 ]['NbLabels']= 3
    BaseLogPartFctRef[ 378 ]['NbCliques']= 137
    BaseLogPartFctRef[ 378 ]['NbSites']= 90
    BaseLogPartFctRef[ 378 ]['StdNgbhDivMoyNgbh']= 0.407746584779
    BaseLogPartFctRef[379 ]={}
    BaseLogPartFctRef[379]['LogPF']=np.array([51.63,52.93,54.28,55.68,57.12,58.61,60.17,61.82,63.49,65.20,67.05,68.93,70.87,72.87,74.96,77.12,79.37,81.78,84.48,87.16,89.92,92.77,95.83,98.92,102.1,105.4,108.7,112.0,115.6,119.0])
    BaseLogPartFctRef[ 379 ]['NbLabels']= 3
    BaseLogPartFctRef[ 379 ]['NbCliques']= 77
    BaseLogPartFctRef[ 379 ]['NbSites']= 47
    BaseLogPartFctRef[ 379 ]['StdNgbhDivMoyNgbh']= 0.346934416658
    BaseLogPartFctRef[380 ]={}
    BaseLogPartFctRef[380]['LogPF']=np.array([36.25,37.00,37.78,38.62,39.51,40.39,41.31,42.27,43.25,44.29,45.32,46.45,47.58,48.78,49.94,51.18,52.49,53.91,55.23,56.72,58.17,59.75,61.39,63.03,64.66,66.42,68.23,69.97,71.83,73.79])
    BaseLogPartFctRef[ 380 ]['NbLabels']= 3
    BaseLogPartFctRef[ 380 ]['NbCliques']= 46
    BaseLogPartFctRef[ 380 ]['NbSites']= 33
    BaseLogPartFctRef[ 380 ]['StdNgbhDivMoyNgbh']= 0.370858752796
    BaseLogPartFctRef[381 ]={}
    BaseLogPartFctRef[381]['LogPF']=np.array([34.06,34.81,35.62,36.43,37.26,38.13,39.02,39.93,40.85,41.84,42.90,43.97,45.06,46.22,47.35,48.58,49.87,51.19,52.62,54.01,55.53,57.08,58.70,60.45,62.20,63.93,65.73,67.53,69.41,71.40])
    BaseLogPartFctRef[ 381 ]['NbLabels']= 3
    BaseLogPartFctRef[ 381 ]['NbCliques']= 45
    BaseLogPartFctRef[ 381 ]['NbSites']= 31
    BaseLogPartFctRef[ 381 ]['StdNgbhDivMoyNgbh']= 0.409167690117
    BaseLogPartFctRef[382 ]={}
    BaseLogPartFctRef[382]['LogPF']=np.array([35.16,35.88,36.63,37.39,38.21,39.03,39.91,40.77,41.67,42.62,43.64,44.69,45.75,46.88,48.05,49.23,50.43,51.73,53.02,54.41,55.86,57.31,58.82,60.44,62.04,63.77,65.46,67.19,68.94,70.75])
    BaseLogPartFctRef[ 382 ]['NbLabels']= 3
    BaseLogPartFctRef[ 382 ]['NbCliques']= 43
    BaseLogPartFctRef[ 382 ]['NbSites']= 32
    BaseLogPartFctRef[ 382 ]['StdNgbhDivMoyNgbh']= 0.407396352413
    BaseLogPartFctRef[383 ]={}
    BaseLogPartFctRef[383]['LogPF']=np.array([50.54,51.54,52.59,53.65,54.78,55.95,57.15,58.41,59.70,61.00,62.39,63.79,65.25,66.76,68.29,69.94,71.61,73.26,75.07,76.99,78.84,80.80,82.91,84.98,87.19,89.43,91.74,94.09,96.48,98.88])
    BaseLogPartFctRef[ 383 ]['NbLabels']= 3
    BaseLogPartFctRef[ 383 ]['NbCliques']= 60
    BaseLogPartFctRef[ 383 ]['NbSites']= 46
    BaseLogPartFctRef[ 383 ]['StdNgbhDivMoyNgbh']= 0.417276648654
    BaseLogPartFctRef[384 ]={}
    BaseLogPartFctRef[384]['LogPF']=np.array([36.25,37.14,38.06,39.05,40.01,41.02,42.09,43.16,44.28,45.40,46.59,47.83,49.22,50.58,52.02,53.47,55.09,56.86,58.73,60.73,62.68,64.65,66.74,68.90,71.10,73.26,75.61,77.96,80.31,82.71])
    BaseLogPartFctRef[ 384 ]['NbLabels']= 3
    BaseLogPartFctRef[ 384 ]['NbCliques']= 53
    BaseLogPartFctRef[ 384 ]['NbSites']= 33
    BaseLogPartFctRef[ 384 ]['StdNgbhDivMoyNgbh']= 0.398545837133
    BaseLogPartFctRef[385 ]={}
    BaseLogPartFctRef[385]['LogPF']=np.array([47.24,48.18,49.15,50.18,51.23,52.32,53.49,54.67,55.89,57.15,58.46,59.81,61.20,62.63,64.17,65.71,67.29,68.95,70.64,72.30,74.08,75.95,77.88,79.88,81.93,83.98,86.14,88.33,90.59,92.91])
    BaseLogPartFctRef[ 385 ]['NbLabels']= 3
    BaseLogPartFctRef[ 385 ]['NbCliques']= 57
    BaseLogPartFctRef[ 385 ]['NbSites']= 43
    BaseLogPartFctRef[ 385 ]['StdNgbhDivMoyNgbh']= 0.345278070397
    BaseLogPartFctRef[386 ]={}
    BaseLogPartFctRef[386]['LogPF']=np.array([48.34,49.42,50.57,51.76,52.99,54.25,55.56,56.89,58.28,59.72,61.21,62.73,64.36,66.00,67.85,69.61,71.54,73.53,75.61,77.82,80.07,82.35,84.82,87.16,89.76,92.33,94.97,97.78,100.7,103.5])
    BaseLogPartFctRef[ 386 ]['NbLabels']= 3
    BaseLogPartFctRef[ 386 ]['NbCliques']= 65
    BaseLogPartFctRef[ 386 ]['NbSites']= 44
    BaseLogPartFctRef[ 386 ]['StdNgbhDivMoyNgbh']= 0.410925150947
    BaseLogPartFctRef[387 ]={}
    BaseLogPartFctRef[387]['LogPF']=np.array([28.56,29.15,29.75,30.38,31.01,31.68,32.36,33.07,33.79,34.54,35.34,36.12,36.92,37.79,38.69,39.64,40.62,41.59,42.57,43.57,44.59,45.67,46.83,48.00,49.29,50.50,51.72,52.99,54.32,55.66])
    BaseLogPartFctRef[ 387 ]['NbLabels']= 3
    BaseLogPartFctRef[ 387 ]['NbCliques']= 34
    BaseLogPartFctRef[ 387 ]['NbSites']= 26
    BaseLogPartFctRef[ 387 ]['StdNgbhDivMoyNgbh']= 0.323685609227
    BaseLogPartFctRef[388 ]={}
    BaseLogPartFctRef[388]['LogPF']=np.array([46.14,47.17,48.23,49.32,50.46,51.62,52.84,54.10,55.35,56.71,58.15,59.66,61.15,62.72,64.31,65.97,67.69,69.65,71.55,73.47,75.60,77.79,80.02,82.35,84.83,87.33,89.76,92.35,94.89,97.53])
    BaseLogPartFctRef[ 388 ]['NbLabels']= 3
    BaseLogPartFctRef[ 388 ]['NbCliques']= 61
    BaseLogPartFctRef[ 388 ]['NbSites']= 42
    BaseLogPartFctRef[ 388 ]['StdNgbhDivMoyNgbh']= 0.438847523211
    BaseLogPartFctRef[389 ]={}
    BaseLogPartFctRef[389]['LogPF']=np.array([60.42,61.70,63.00,64.34,65.71,67.20,68.64,70.16,71.77,73.35,74.97,76.66,78.52,80.33,82.13,84.36,86.35,88.49,90.66,92.96,95.15,97.76,100.5,103.1,105.7,108.7,111.7,114.7,117.8,120.9])
    BaseLogPartFctRef[ 389 ]['NbLabels']= 3
    BaseLogPartFctRef[ 389 ]['NbCliques']= 74
    BaseLogPartFctRef[ 389 ]['NbSites']= 55
    BaseLogPartFctRef[ 389 ]['StdNgbhDivMoyNgbh']= 0.404940094816
    BaseLogPartFctRef[390 ]={}
    BaseLogPartFctRef[390]['LogPF']=np.array([40.65,41.46,42.30,43.15,44.04,44.97,45.91,46.92,47.95,49.04,50.10,51.24,52.38,53.58,54.85,56.14,57.44,58.81,60.24,61.66,63.15,64.75,66.43,68.17,69.91,71.71,73.55,75.41,77.31,79.25])
    BaseLogPartFctRef[ 390 ]['NbLabels']= 3
    BaseLogPartFctRef[ 390 ]['NbCliques']= 48
    BaseLogPartFctRef[ 390 ]['NbSites']= 37
    BaseLogPartFctRef[ 390 ]['StdNgbhDivMoyNgbh']= 0.448472572091
    BaseLogPartFctRef[391 ]={}
    BaseLogPartFctRef[391]['LogPF']=np.array([39.55,40.31,41.09,41.92,42.75,43.62,44.50,45.43,46.37,47.34,48.38,49.47,50.60,51.74,52.90,54.11,55.33,56.65,57.92,59.28,60.72,62.14,63.61,65.08,66.62,68.21,69.83,71.49,73.21,74.98])
    BaseLogPartFctRef[ 391 ]['NbLabels']= 3
    BaseLogPartFctRef[ 391 ]['NbCliques']= 45
    BaseLogPartFctRef[ 391 ]['NbSites']= 36
    BaseLogPartFctRef[ 391 ]['StdNgbhDivMoyNgbh']= 0.324074074074
    BaseLogPartFctRef[392 ]={}
    BaseLogPartFctRef[392]['LogPF']=np.array([28.56,29.12,29.69,30.29,30.92,31.57,32.22,32.88,33.57,34.30,35.04,35.82,36.62,37.41,38.30,39.16,40.04,40.97,41.96,43.00,44.01,45.06,46.19,47.30,48.50,49.67,50.90,52.13,53.45,54.76])
    BaseLogPartFctRef[ 392 ]['NbLabels']= 3
    BaseLogPartFctRef[ 392 ]['NbCliques']= 33
    BaseLogPartFctRef[ 392 ]['NbSites']= 26
    BaseLogPartFctRef[ 392 ]['StdNgbhDivMoyNgbh']= 0.319185639572
    BaseLogPartFctRef[393 ]={}
    BaseLogPartFctRef[393]['LogPF']=np.array([62.62,64.15,65.75,67.37,69.06,70.77,72.51,74.38,76.36,78.34,80.38,82.36,84.41,86.77,89.12,91.65,94.23,96.95,99.79,102.9,106.0,109.5,113.0,116.8,120.4,124.4,128.5,132.5,136.6,140.8])
    BaseLogPartFctRef[ 393 ]['NbLabels']= 3
    BaseLogPartFctRef[ 393 ]['NbCliques']= 91
    BaseLogPartFctRef[ 393 ]['NbSites']= 57
    BaseLogPartFctRef[ 393 ]['StdNgbhDivMoyNgbh']= 0.361955295314
    BaseLogPartFctRef[394 ]={}
    BaseLogPartFctRef[394]['LogPF']=np.array([51.63,52.79,53.99,55.20,56.44,57.73,59.06,60.47,61.88,63.36,64.90,66.51,68.16,69.83,71.62,73.54,75.46,77.41,79.53,81.70,83.96,86.38,88.72,91.15,93.74,96.27,99.02,101.8,104.5,107.4])
    BaseLogPartFctRef[ 394 ]['NbLabels']= 3
    BaseLogPartFctRef[ 394 ]['NbCliques']= 67
    BaseLogPartFctRef[ 394 ]['NbSites']= 47
    BaseLogPartFctRef[ 394 ]['StdNgbhDivMoyNgbh']= 0.406664275469
    BaseLogPartFctRef[395 ]={}
    BaseLogPartFctRef[395]['LogPF']=np.array([49.44,50.53,51.64,52.82,54.01,55.24,56.48,57.81,59.25,60.68,62.10,63.64,65.22,66.87,68.57,70.33,72.06,73.93,75.99,78.07,80.20,82.38,84.61,87.04,89.48,92.07,94.68,97.28,99.96,102.7])
    BaseLogPartFctRef[ 395 ]['NbLabels']= 3
    BaseLogPartFctRef[ 395 ]['NbCliques']= 64
    BaseLogPartFctRef[ 395 ]['NbSites']= 45
    BaseLogPartFctRef[ 395 ]['StdNgbhDivMoyNgbh']= 0.389957608886
    BaseLogPartFctRef[396 ]={}
    BaseLogPartFctRef[396]['LogPF']=np.array([43.94,45.08,46.20,47.35,48.56,49.77,51.03,52.34,53.71,55.15,56.62,58.16,59.76,61.44,63.13,64.89,66.86,68.91,71.15,73.28,75.53,77.93,80.42,82.88,85.60,88.37,91.23,94.02,96.82,99.69])
    BaseLogPartFctRef[ 396 ]['NbLabels']= 3
    BaseLogPartFctRef[ 396 ]['NbCliques']= 64
    BaseLogPartFctRef[ 396 ]['NbSites']= 40
    BaseLogPartFctRef[ 396 ]['StdNgbhDivMoyNgbh']= 0.386605096936
    BaseLogPartFctRef[397 ]={}
    BaseLogPartFctRef[397]['LogPF']=np.array([54.93,56.00,57.14,58.31,59.45,60.72,62.02,63.29,64.62,66.07,67.50,68.99,70.43,72.02,73.64,75.42,77.15,78.99,80.83,82.72,84.45,86.30,88.33,90.62,92.97,95.32,97.73,100.1,102.4,104.8])
    BaseLogPartFctRef[ 397 ]['NbLabels']= 3
    BaseLogPartFctRef[ 397 ]['NbCliques']= 64
    BaseLogPartFctRef[ 397 ]['NbSites']= 50
    BaseLogPartFctRef[ 397 ]['StdNgbhDivMoyNgbh']= 0.384035546247
    BaseLogPartFctRef[398 ]={}
    BaseLogPartFctRef[398]['LogPF']=np.array([49.44,50.34,51.28,52.24,53.27,54.33,55.38,56.48,57.64,58.84,60.09,61.38,62.68,64.04,65.39,66.78,68.28,69.80,71.33,72.96,74.60,76.30,78.09,79.84,81.71,83.62,85.60,87.63,89.59,91.66])
    BaseLogPartFctRef[ 398 ]['NbLabels']= 3
    BaseLogPartFctRef[ 398 ]['NbCliques']= 54
    BaseLogPartFctRef[ 398 ]['NbSites']= 45
    BaseLogPartFctRef[ 398 ]['StdNgbhDivMoyNgbh']= 0.387929445501
    BaseLogPartFctRef[399 ]={}
    BaseLogPartFctRef[399]['LogPF']=np.array([76.90,78.46,80.09,81.77,83.51,85.39,87.25,89.14,91.11,93.21,95.36,97.66,100.1,102.6,105.1,107.6,110.3,113.1,116.0,118.8,121.6,124.8,128.1,131.5,135.0,138.6,142.0,145.6,149.3,152.9])
    BaseLogPartFctRef[ 399 ]['NbLabels']= 3
    BaseLogPartFctRef[ 399 ]['NbCliques']= 95
    BaseLogPartFctRef[ 399 ]['NbSites']= 70
    BaseLogPartFctRef[ 399 ]['StdNgbhDivMoyNgbh']= 0.369039376788
    BaseLogPartFctRef[400 ]={}
    BaseLogPartFctRef[400]['LogPF']=np.array([64.82,66.29,67.82,69.35,70.93,72.55,74.29,76.10,77.90,79.79,81.64,83.70,85.96,88.21,90.46,92.78,95.13,97.77,100.2,103.0,105.8,108.7,111.9,115.0,118.4,121.7,125.0,128.5,131.8,135.4])
    BaseLogPartFctRef[ 400 ]['NbLabels']= 3
    BaseLogPartFctRef[ 400 ]['NbCliques']= 86
    BaseLogPartFctRef[ 400 ]['NbSites']= 59
    BaseLogPartFctRef[ 400 ]['StdNgbhDivMoyNgbh']= 0.358206345182
    BaseLogPartFctRef[401 ]={}
    BaseLogPartFctRef[401]['LogPF']=np.array([73.61,75.16,76.75,78.41,80.19,81.91,83.69,85.65,87.56,89.45,91.46,93.83,96.00,98.55,101.0,103.5,106.0,108.7,111.6,114.4,117.3,120.5,123.8,127.0,130.5,133.8,137.6,141.3,144.8,148.6])
    BaseLogPartFctRef[ 401 ]['NbLabels']= 3
    BaseLogPartFctRef[ 401 ]['NbCliques']= 93
    BaseLogPartFctRef[ 401 ]['NbSites']= 67
    BaseLogPartFctRef[ 401 ]['StdNgbhDivMoyNgbh']= 0.377028770411
    BaseLogPartFctRef[402 ]={}
    BaseLogPartFctRef[402]['LogPF']=np.array([38.45,39.21,40.00,40.84,41.68,42.56,43.47,44.40,45.35,46.35,47.34,48.39,49.49,50.61,51.76,52.97,54.20,55.53,56.95,58.35,59.81,61.36,62.91,64.46,66.08,67.73,69.45,71.14,72.91,74.74])
    BaseLogPartFctRef[ 402 ]['NbLabels']= 3
    BaseLogPartFctRef[ 402 ]['NbCliques']= 45
    BaseLogPartFctRef[ 402 ]['NbSites']= 35
    BaseLogPartFctRef[ 402 ]['StdNgbhDivMoyNgbh']= 0.3861653904
    BaseLogPartFctRef[403 ]={}
    BaseLogPartFctRef[403]['LogPF']=np.array([34.06,34.76,35.46,36.20,36.99,37.76,38.59,39.42,40.29,41.19,42.14,43.13,44.11,45.12,46.23,47.34,48.51,49.76,51.02,52.32,53.69,55.11,56.55,58.03,59.60,61.18,62.85,64.51,66.21,67.92])
    BaseLogPartFctRef[ 403 ]['NbLabels']= 3
    BaseLogPartFctRef[ 403 ]['NbCliques']= 41
    BaseLogPartFctRef[ 403 ]['NbSites']= 31
    BaseLogPartFctRef[ 403 ]['StdNgbhDivMoyNgbh']= 0.478807146541
    BaseLogPartFctRef[404 ]={}
    BaseLogPartFctRef[404]['LogPF']=np.array([53.83,55.20,56.57,58.01,59.45,60.95,62.56,64.17,65.88,67.65,69.48,71.35,73.29,75.31,77.53,79.74,82.01,84.41,87.14,89.82,92.62,95.69,98.77,101.9,105.2,108.6,112.0,115.5,119.1,122.6])
    BaseLogPartFctRef[ 404 ]['NbLabels']= 3
    BaseLogPartFctRef[ 404 ]['NbCliques']= 79
    BaseLogPartFctRef[ 404 ]['NbSites']= 49
    BaseLogPartFctRef[ 404 ]['StdNgbhDivMoyNgbh']= 0.389463364081

    #In [7]: ComputeBaseLogPartFctRef_NonReg(FirstIndex=(255+150),NbExtraIndex=30,BetaMax=1.45,DeltaBeta=0.05,NbLabels=3,NBX=15,NBY=15,NBZ=15,BetaGeneration=0.3)
    BaseLogPartFctRef[405 ]={}
    BaseLogPartFctRef[405]['LogPF']=np.array([508.7,523.6,538.6,554.7,570.8,587.1,604.1,621.4,640.4,660.1,681.2,702.8,728.3,753.8,783.3,811.6,842.0,875.1,908.3,944.1,980.3,1016.5,1053.4,1091.6,1131.5,1171.4,1211.5,1250.7,1290.6,1331.0])
    BaseLogPartFctRef[ 405 ]['NbLabels']= 3
    BaseLogPartFctRef[ 405 ]['NbCliques']= 863
    BaseLogPartFctRef[ 405 ]['NbSites']= 463
    BaseLogPartFctRef[ 405 ]['StdNgbhDivMoyNgbh']= 0.351784832347
    BaseLogPartFctRef[406 ]={}
    BaseLogPartFctRef[406]['LogPF']=np.array([738.3,760.5,782.7,806.4,830.6,855.2,881.3,907.9,936.5,967.3,998.2,1030.5,1067.6,1106.7,1148.7,1192.9,1240.5,1289.2,1341.6,1394.4,1449.9,1507.0,1565.2,1624.1,1684.5,1744.5,1805.6,1867.4,1930.2,1992.1])
    BaseLogPartFctRef[ 406 ]['NbLabels']= 3
    BaseLogPartFctRef[ 406 ]['NbCliques']= 1299
    BaseLogPartFctRef[ 406 ]['NbSites']= 672
    BaseLogPartFctRef[ 406 ]['StdNgbhDivMoyNgbh']= 0.33520591512
    BaseLogPartFctRef[407 ]={}
    BaseLogPartFctRef[407]['LogPF']=np.array([195.6,200.8,206.2,211.6,217.4,223.4,229.7,236.0,242.6,249.3,256.3,263.7,271.3,279.4,287.8,296.8,306.9,317.1,327.3,338.5,349.6,361.5,374.2,387.3,400.7,414.0,427.6,441.6,455.2,469.2])
    BaseLogPartFctRef[ 407 ]['NbLabels']= 3
    BaseLogPartFctRef[ 407 ]['NbCliques']= 306
    BaseLogPartFctRef[ 407 ]['NbSites']= 178
    BaseLogPartFctRef[ 407 ]['StdNgbhDivMoyNgbh']= 0.377850658458
    BaseLogPartFctRef[408 ]={}
    BaseLogPartFctRef[408]['LogPF']=np.array([522.9,537.2,552.1,567.6,583.3,599.7,617.0,634.7,653.3,671.9,692.6,714.9,736.2,761.5,788.9,817.2,845.1,875.4,908.1,942.4,976.8,1013.6,1049.7,1087.4,1126.1,1164.3,1202.8,1242.2,1281.7,1322.0])
    BaseLogPartFctRef[ 408 ]['NbLabels']= 3
    BaseLogPartFctRef[ 408 ]['NbCliques']= 859
    BaseLogPartFctRef[ 408 ]['NbSites']= 476
    BaseLogPartFctRef[ 408 ]['StdNgbhDivMoyNgbh']= 0.347821584232
    BaseLogPartFctRef[409 ]={}
    BaseLogPartFctRef[409]['LogPF']=np.array([551.5,566.4,581.7,597.7,614.2,630.9,648.6,667.5,687.0,706.8,728.1,749.6,773.2,798.2,824.4,853.3,882.0,913.2,946.0,980.2,1013.0,1049.1,1086.2,1123.2,1161.0,1199.8,1240.2,1280.0,1320.1,1360.7])
    BaseLogPartFctRef[ 409 ]['NbLabels']= 3
    BaseLogPartFctRef[ 409 ]['NbCliques']= 883
    BaseLogPartFctRef[ 409 ]['NbSites']= 502
    BaseLogPartFctRef[ 409 ]['StdNgbhDivMoyNgbh']= 0.356185598361
    BaseLogPartFctRef[410 ]={}
    BaseLogPartFctRef[410]['LogPF']=np.array([927.2,955.0,983.8,1013.0,1043.9,1074.9,1106.5,1139.9,1174.8,1212.6,1254.0,1296.0,1339.7,1389.9,1440.6,1496.7,1555.2,1616.5,1683.7,1752.2,1820.8,1889.9,1961.5,2033.1,2108.2,2181.8,2256.7,2332.7,2408.5,2485.0])
    BaseLogPartFctRef[ 410 ]['NbLabels']= 3
    BaseLogPartFctRef[ 410 ]['NbCliques']= 1602
    BaseLogPartFctRef[ 410 ]['NbSites']= 844
    BaseLogPartFctRef[ 410 ]['StdNgbhDivMoyNgbh']= 0.330651445748
    BaseLogPartFctRef[411 ]={}
    BaseLogPartFctRef[411]['LogPF']=np.array([432.9,444.3,456.5,468.7,481.7,494.7,508.2,522.3,538.1,553.1,568.7,585.2,603.3,621.5,641.0,661.0,684.2,709.9,737.3,763.5,791.6,818.2,846.4,876.5,906.1,937.2,968.5,999.4,1030.5,1062.5])
    BaseLogPartFctRef[ 411 ]['NbLabels']= 3
    BaseLogPartFctRef[ 411 ]['NbCliques']= 688
    BaseLogPartFctRef[ 411 ]['NbSites']= 394
    BaseLogPartFctRef[ 411 ]['StdNgbhDivMoyNgbh']= 0.387826059151
    BaseLogPartFctRef[412 ]={}
    BaseLogPartFctRef[412]['LogPF']=np.array([189.0,193.6,198.4,203.4,208.6,213.9,219.4,225.2,231.3,237.3,243.7,250.3,257.4,264.9,272.0,280.1,288.7,297.2,306.1,315.8,326.2,336.6,347.3,358.3,370.0,381.8,394.0,406.3,418.6,431.1])
    BaseLogPartFctRef[ 412 ]['NbLabels']= 3
    BaseLogPartFctRef[ 412 ]['NbCliques']= 279
    BaseLogPartFctRef[ 412 ]['NbSites']= 172
    BaseLogPartFctRef[ 412 ]['StdNgbhDivMoyNgbh']= 0.373680618184
    BaseLogPartFctRef[413 ]={}
    BaseLogPartFctRef[413]['LogPF']=np.array([512.0,526.0,540.3,554.7,570.0,585.9,602.9,620.0,637.5,656.9,675.9,696.5,717.4,740.2,765.5,790.2,818.8,849.8,881.9,913.0,947.5,981.7,1016.8,1052.4,1088.2,1125.4,1162.4,1199.6,1238.6,1277.1])
    BaseLogPartFctRef[ 413 ]['NbLabels']= 3
    BaseLogPartFctRef[ 413 ]['NbCliques']= 825
    BaseLogPartFctRef[ 413 ]['NbSites']= 466
    BaseLogPartFctRef[ 413 ]['StdNgbhDivMoyNgbh']= 0.376500894395
    BaseLogPartFctRef[414 ]={}
    BaseLogPartFctRef[414]['LogPF']=np.array([206.5,212.2,218.1,224.0,230.1,236.4,242.7,249.5,256.4,263.5,271.3,279.2,287.4,295.9,305.0,314.4,324.5,335.0,346.0,357.9,369.9,382.6,395.8,409.6,423.1,437.0,451.7,466.6,481.8,496.8])
    BaseLogPartFctRef[ 414 ]['NbLabels']= 3
    BaseLogPartFctRef[ 414 ]['NbCliques']= 325
    BaseLogPartFctRef[ 414 ]['NbSites']= 188
    BaseLogPartFctRef[ 414 ]['StdNgbhDivMoyNgbh']= 0.347733632011
    BaseLogPartFctRef[415 ]={}
    BaseLogPartFctRef[415]['LogPF']=np.array([370.2,380.2,390.2,400.8,411.8,423.0,434.6,446.9,459.3,472.1,485.9,500.2,515.4,531.2,548.4,567.2,586.4,605.6,626.5,647.6,669.6,693.2,717.3,741.0,766.4,791.7,817.7,843.9,870.2,896.8])
    BaseLogPartFctRef[ 415 ]['NbLabels']= 3
    BaseLogPartFctRef[ 415 ]['NbCliques']= 579
    BaseLogPartFctRef[ 415 ]['NbSites']= 337
    BaseLogPartFctRef[ 415 ]['StdNgbhDivMoyNgbh']= 0.368270578511
    BaseLogPartFctRef[416 ]={}
    BaseLogPartFctRef[416]['LogPF']=np.array([466.9,480.1,493.9,507.8,522.4,537.4,552.7,569.0,585.7,603.0,621.1,640.8,661.5,683.7,709.4,732.9,760.5,789.6,821.0,853.3,886.0,918.9,952.1,987.2,1021.7,1056.8,1093.6,1130.1,1166.2,1202.7])
    BaseLogPartFctRef[ 416 ]['NbLabels']= 3
    BaseLogPartFctRef[ 416 ]['NbCliques']= 780
    BaseLogPartFctRef[ 416 ]['NbSites']= 425
    BaseLogPartFctRef[ 416 ]['StdNgbhDivMoyNgbh']= 0.355694695333
    BaseLogPartFctRef[417 ]={}
    BaseLogPartFctRef[417]['LogPF']=np.array([1316.1,1355.1,1395.9,1437.3,1479.8,1523.6,1570.1,1618.1,1666.9,1722.6,1779.0,1838.6,1900.8,1968.1,2038.2,2117.3,2201.7,2289.6,2383.0,2477.3,2578.4,2679.5,2783.4,2885.1,2990.8,3097.2,3204.4,3312.3,3420.5,3530.0])
    BaseLogPartFctRef[ 417 ]['NbLabels']= 3
    BaseLogPartFctRef[ 417 ]['NbCliques']= 2283
    BaseLogPartFctRef[ 417 ]['NbSites']= 1198
    BaseLogPartFctRef[ 417 ]['StdNgbhDivMoyNgbh']= 0.329917495293
    BaseLogPartFctRef[418 ]={}
    BaseLogPartFctRef[418]['LogPF']=np.array([329.6,338.8,348.1,357.9,368.3,379.4,390.0,401.8,413.6,425.9,438.5,452.0,466.8,482.0,497.8,516.7,535.6,556.5,574.5,596.1,618.3,640.6,664.2,688.1,712.2,736.9,761.7,786.7,812.0,837.8])
    BaseLogPartFctRef[ 418 ]['NbLabels']= 3
    BaseLogPartFctRef[ 418 ]['NbCliques']= 546
    BaseLogPartFctRef[ 418 ]['NbSites']= 300
    BaseLogPartFctRef[ 418 ]['StdNgbhDivMoyNgbh']= 0.363634442206
    BaseLogPartFctRef[419 ]={}
    BaseLogPartFctRef[419]['LogPF']=np.array([483.4,495.9,508.6,522.6,536.8,551.3,566.0,581.0,597.3,613.4,630.8,648.2,666.5,686.2,707.4,731.8,756.4,780.4,807.6,835.7,864.6,894.2,924.2,956.0,988.6,1021.2,1053.7,1087.2,1119.8,1153.7])
    BaseLogPartFctRef[ 419 ]['NbLabels']= 3
    BaseLogPartFctRef[ 419 ]['NbCliques']= 741
    BaseLogPartFctRef[ 419 ]['NbSites']= 440
    BaseLogPartFctRef[ 419 ]['StdNgbhDivMoyNgbh']= 0.377967091594
    BaseLogPartFctRef[420 ]={}
    BaseLogPartFctRef[420]['LogPF']=np.array([565.8,581.5,597.8,614.3,632.0,649.8,668.4,687.5,708.1,728.4,751.1,775.3,801.0,829.1,857.6,885.8,918.2,951.8,988.2,1023.6,1061.1,1100.1,1138.6,1179.8,1220.4,1261.5,1303.5,1345.9,1388.7,1431.6])
    BaseLogPartFctRef[ 420 ]['NbLabels']= 3
    BaseLogPartFctRef[ 420 ]['NbCliques']= 926
    BaseLogPartFctRef[ 420 ]['NbSites']= 515
    BaseLogPartFctRef[ 420 ]['StdNgbhDivMoyNgbh']= 0.368924387478
    BaseLogPartFctRef[421 ]={}
    BaseLogPartFctRef[421]['LogPF']=np.array([672.4,691.0,711.1,731.8,752.7,774.0,797.0,819.8,845.4,871.2,897.1,924.9,956.1,989.2,1026.3,1066.4,1106.1,1147.0,1191.4,1237.7,1285.0,1334.0,1382.6,1432.9,1483.7,1535.2,1587.3,1639.8,1692.8,1745.9])
    BaseLogPartFctRef[ 421 ]['NbLabels']= 3
    BaseLogPartFctRef[ 421 ]['NbCliques']= 1128
    BaseLogPartFctRef[ 421 ]['NbSites']= 612
    BaseLogPartFctRef[ 421 ]['StdNgbhDivMoyNgbh']= 0.354709146661
    BaseLogPartFctRef[422 ]={}
    BaseLogPartFctRef[422]['LogPF']=np.array([588.9,604.9,621.8,639.7,658.0,676.1,695.9,716.0,735.9,757.0,780.8,804.8,832.4,860.7,890.8,922.0,956.8,994.2,1032.0,1071.2,1111.3,1153.3,1195.5,1238.8,1283.1,1326.5,1371.7,1417.5,1463.3,1509.1])
    BaseLogPartFctRef[ 422 ]['NbLabels']= 3
    BaseLogPartFctRef[ 422 ]['NbCliques']= 978
    BaseLogPartFctRef[ 422 ]['NbSites']= 536
    BaseLogPartFctRef[ 422 ]['StdNgbhDivMoyNgbh']= 0.351271838626
    BaseLogPartFctRef[423 ]={}
    BaseLogPartFctRef[423]['LogPF']=np.array([586.7,603.2,619.5,637.1,654.6,673.5,692.6,712.6,733.6,754.2,777.0,800.9,825.9,852.5,879.6,910.3,943.8,977.1,1013.6,1052.4,1090.7,1132.2,1173.0,1214.9,1257.9,1301.0,1344.7,1388.1,1431.7,1476.2])
    BaseLogPartFctRef[ 423 ]['NbLabels']= 3
    BaseLogPartFctRef[ 423 ]['NbCliques']= 956
    BaseLogPartFctRef[ 423 ]['NbSites']= 534
    BaseLogPartFctRef[ 423 ]['StdNgbhDivMoyNgbh']= 0.367098803098
    BaseLogPartFctRef[424 ]={}
    BaseLogPartFctRef[424]['LogPF']=np.array([337.3,345.7,354.3,363.6,373.2,383.1,393.5,404.2,415.3,426.7,438.5,451.1,464.6,478.6,492.8,507.3,521.9,537.5,554.6,573.5,592.7,612.6,631.9,653.4,674.4,696.6,718.6,741.2,764.5,788.3])
    BaseLogPartFctRef[ 424 ]['NbLabels']= 3
    BaseLogPartFctRef[ 424 ]['NbCliques']= 511
    BaseLogPartFctRef[ 424 ]['NbSites']= 307
    BaseLogPartFctRef[ 424 ]['StdNgbhDivMoyNgbh']= 0.344224972374
    BaseLogPartFctRef[425 ]={}
    BaseLogPartFctRef[425]['LogPF']=np.array([448.2,460.6,473.5,487.0,500.3,514.0,528.6,543.8,559.2,575.4,592.7,611.4,630.3,650.9,672.1,694.1,719.0,747.5,777.2,805.5,835.9,867.5,898.4,930.5,961.9,993.2,1025.7,1058.0,1090.7,1124.1])
    BaseLogPartFctRef[ 425 ]['NbLabels']= 3
    BaseLogPartFctRef[ 425 ]['NbCliques']= 727
    BaseLogPartFctRef[ 425 ]['NbSites']= 408
    BaseLogPartFctRef[ 425 ]['StdNgbhDivMoyNgbh']= 0.385626302272
    BaseLogPartFctRef[426 ]={}
    BaseLogPartFctRef[426]['LogPF']=np.array([307.6,316.1,324.7,333.0,342.5,352.0,361.7,372.0,382.8,393.6,405.2,416.6,430.9,445.8,461.1,477.4,493.4,512.3,528.5,545.9,565.4,585.7,607.2,628.5,650.3,672.3,694.5,716.4,739.1,762.3])
    BaseLogPartFctRef[ 426 ]['NbLabels']= 3
    BaseLogPartFctRef[ 426 ]['NbCliques']= 493
    BaseLogPartFctRef[ 426 ]['NbSites']= 280
    BaseLogPartFctRef[ 426 ]['StdNgbhDivMoyNgbh']= 0.381837528918
    BaseLogPartFctRef[427 ]={}
    BaseLogPartFctRef[427]['LogPF']=np.array([767.9,788.8,810.4,832.0,855.1,879.6,903.7,930.1,957.2,983.9,1014.0,1043.4,1076.3,1110.2,1149.5,1187.2,1228.3,1270.4,1318.0,1369.0,1418.9,1469.4,1523.3,1574.8,1629.0,1684.0,1741.4,1798.5,1856.3,1914.9])
    BaseLogPartFctRef[ 427 ]['NbLabels']= 3
    BaseLogPartFctRef[ 427 ]['NbCliques']= 1233
    BaseLogPartFctRef[ 427 ]['NbSites']= 699
    BaseLogPartFctRef[ 427 ]['StdNgbhDivMoyNgbh']= 0.346283953722
    BaseLogPartFctRef[428 ]={}
    BaseLogPartFctRef[428]['LogPF']=np.array([557.0,572.9,589.4,606.3,624.2,642.2,660.9,680.6,700.6,722.2,744.5,770.9,798.4,824.6,854.7,886.0,921.5,955.9,993.0,1031.1,1070.2,1110.0,1150.1,1191.1,1232.9,1275.5,1318.4,1361.5,1405.6,1449.7])
    BaseLogPartFctRef[ 428 ]['NbLabels']= 3
    BaseLogPartFctRef[ 428 ]['NbCliques']= 939
    BaseLogPartFctRef[ 428 ]['NbSites']= 507
    BaseLogPartFctRef[ 428 ]['StdNgbhDivMoyNgbh']= 0.356102461985
    BaseLogPartFctRef[429 ]={}
    BaseLogPartFctRef[429]['LogPF']=np.array([332.9,341.1,349.8,358.9,368.0,377.5,387.0,396.9,407.4,418.4,429.3,441.5,454.5,467.6,480.7,494.4,509.2,523.6,539.4,555.9,573.7,592.1,611.1,630.9,651.4,671.1,692.0,714.0,735.9,758.5])
    BaseLogPartFctRef[ 429 ]['NbLabels']= 3
    BaseLogPartFctRef[ 429 ]['NbCliques']= 489
    BaseLogPartFctRef[ 429 ]['NbSites']= 303
    BaseLogPartFctRef[ 429 ]['StdNgbhDivMoyNgbh']= 0.376854102508
    BaseLogPartFctRef[430 ]={}
    BaseLogPartFctRef[430]['LogPF']=np.array([581.2,595.9,611.6,627.8,644.3,661.5,680.1,698.8,718.3,739.5,760.4,783.2,808.9,834.1,860.8,888.1,919.9,950.2,983.1,1018.2,1054.5,1090.4,1128.9,1167.4,1205.6,1246.5,1287.0,1328.0,1368.0,1409.5])
    BaseLogPartFctRef[ 430 ]['NbLabels']= 3
    BaseLogPartFctRef[ 430 ]['NbCliques']= 906
    BaseLogPartFctRef[ 430 ]['NbSites']= 529
    BaseLogPartFctRef[ 430 ]['StdNgbhDivMoyNgbh']= 0.37396580405
    BaseLogPartFctRef[431 ]={}
    BaseLogPartFctRef[431]['LogPF']=np.array([438.3,450.1,462.4,475.4,488.5,502.0,515.5,530.0,544.5,559.7,575.5,593.3,612.5,632.0,653.3,673.8,699.2,723.6,748.3,774.2,801.9,830.4,859.2,888.5,919.7,951.4,983.6,1014.9,1046.8,1079.4])
    BaseLogPartFctRef[ 431 ]['NbLabels']= 3
    BaseLogPartFctRef[ 431 ]['NbCliques']= 700
    BaseLogPartFctRef[ 431 ]['NbSites']= 399
    BaseLogPartFctRef[ 431 ]['StdNgbhDivMoyNgbh']= 0.387604113959
    BaseLogPartFctRef[432 ]={}
    BaseLogPartFctRef[432]['LogPF']=np.array([414.2,426.8,439.8,453.1,466.6,480.6,495.2,511.3,527.2,544.1,563.5,583.1,605.2,627.7,653.3,680.1,707.9,735.1,764.4,796.9,829.7,862.1,895.4,929.8,963.4,998.3,1033.0,1068.0,1103.3,1138.5])
    BaseLogPartFctRef[ 432 ]['NbLabels']= 3
    BaseLogPartFctRef[ 432 ]['NbCliques']= 739
    BaseLogPartFctRef[ 432 ]['NbSites']= 377
    BaseLogPartFctRef[ 432 ]['StdNgbhDivMoyNgbh']= 0.351961232139
    BaseLogPartFctRef[433 ]={}
    BaseLogPartFctRef[433]['LogPF']=np.array([472.4,485.9,499.6,513.8,528.4,543.4,559.0,575.0,591.4,609.2,627.9,647.6,668.6,691.5,715.8,739.4,767.4,795.7,824.0,856.6,888.3,921.9,954.9,988.6,1022.6,1057.7,1093.9,1129.4,1165.7,1201.7])
    BaseLogPartFctRef[ 433 ]['NbLabels']= 3
    BaseLogPartFctRef[ 433 ]['NbCliques']= 778
    BaseLogPartFctRef[ 433 ]['NbSites']= 430
    BaseLogPartFctRef[ 433 ]['StdNgbhDivMoyNgbh']= 0.356626926853
    BaseLogPartFctRef[434 ]={}
    BaseLogPartFctRef[434]['LogPF']=np.array([640.5,659.2,678.7,698.1,718.2,739.3,761.1,783.4,806.8,830.9,855.9,883.8,916.8,947.4,983.6,1018.9,1058.7,1100.9,1144.3,1188.8,1234.1,1280.0,1328.0,1376.1,1425.7,1474.4,1524.0,1573.2,1623.7,1674.4])
    BaseLogPartFctRef[ 434 ]['NbLabels']= 3
    BaseLogPartFctRef[ 434 ]['NbCliques']= 1083
    BaseLogPartFctRef[ 434 ]['NbSites']= 583
    BaseLogPartFctRef[ 434 ]['StdNgbhDivMoyNgbh']= 0.362779359663

    #In [8]: ComputeBaseLogPartFctRef_NonReg(FirstIndex=(255+180),NbExtraIndex=30,BetaMax=1.45,DeltaBeta=0.05,NbLabels=3,NBX=15,NBY=15,NBZ=15,BetaGeneration=0.5)
    BaseLogPartFctRef[435 ]={}
    BaseLogPartFctRef[435]['LogPF']=np.array([3246.4,3372.1,3503.8,3639.8,3778.6,3926.2,4081.7,4244.7,4424.1,4614.2,4838.0,5097.4,5388.1,5699.5,6018.5,6356.4,6703.3,7051.5,7405.2,7761.2,8123.4,8485.1,8848.1,9216.1,9584.6,9953.5,10322.0,10693.0,11064.9,11436.4])
    BaseLogPartFctRef[ 435 ]['NbLabels']= 3
    BaseLogPartFctRef[ 435 ]['NbCliques']= 7487
    BaseLogPartFctRef[ 435 ]['NbSites']= 2955
    BaseLogPartFctRef[ 435 ]['StdNgbhDivMoyNgbh']= 0.190295523564
    BaseLogPartFctRef[436 ]={}
    BaseLogPartFctRef[436]['LogPF']=np.array([3299.1,3430.6,3566.7,3706.0,3848.8,4000.4,4158.9,4323.0,4506.5,4713.0,4964.8,5240.2,5547.9,5874.9,6212.9,6558.0,6913.2,7271.6,7642.4,8015.9,8391.9,8769.6,9147.9,9528.3,9910.8,10293.5,10677.1,11062.5,11447.8,11834.2])
    BaseLogPartFctRef[ 436 ]['NbLabels']= 3
    BaseLogPartFctRef[ 436 ]['NbCliques']= 7755
    BaseLogPartFctRef[ 436 ]['NbSites']= 3003
    BaseLogPartFctRef[ 436 ]['StdNgbhDivMoyNgbh']= 0.177143540895
    BaseLogPartFctRef[437 ]={}
    BaseLogPartFctRef[437]['LogPF']=np.array([3228.8,3354.9,3484.2,3620.1,3761.2,3908.5,4060.7,4222.6,4397.8,4586.3,4806.9,5076.8,5369.9,5678.4,6010.0,6356.2,6707.0,7055.3,7409.4,7767.3,8130.5,8497.9,8865.2,9234.5,9604.5,9975.2,10347.6,10720.5,11093.2,11467.1])
    BaseLogPartFctRef[ 437 ]['NbLabels']= 3
    BaseLogPartFctRef[ 437 ]['NbCliques']= 7520
    BaseLogPartFctRef[ 437 ]['NbSites']= 2939
    BaseLogPartFctRef[ 437 ]['StdNgbhDivMoyNgbh']= 0.190823564003
    BaseLogPartFctRef[438 ]={}
    BaseLogPartFctRef[438]['LogPF']=np.array([3173.9,3296.5,3423.2,3555.4,3690.7,3831.4,3978.9,4137.3,4311.5,4507.1,4723.6,4979.9,5251.6,5536.4,5844.6,6171.9,6510.2,6852.1,7197.1,7547.5,7897.5,8252.1,8608.4,8967.4,9326.2,9686.4,10047.6,10408.7,10770.4,11132.8])
    BaseLogPartFctRef[ 438 ]['NbLabels']= 3
    BaseLogPartFctRef[ 438 ]['NbCliques']= 7298
    BaseLogPartFctRef[ 438 ]['NbSites']= 2889
    BaseLogPartFctRef[ 438 ]['StdNgbhDivMoyNgbh']= 0.188896482419
    BaseLogPartFctRef[439 ]={}
    BaseLogPartFctRef[439]['LogPF']=np.array([3234.3,3360.1,3491.4,3626.2,3767.1,3911.2,4063.2,4231.0,4400.4,4602.5,4845.9,5110.4,5401.3,5709.5,6034.0,6371.5,6712.4,7066.9,7423.3,7784.4,8147.7,8512.0,8879.5,9247.3,9616.6,9986.5,10358.8,10731.7,11104.2,11477.8])
    BaseLogPartFctRef[ 439 ]['NbLabels']= 3
    BaseLogPartFctRef[ 439 ]['NbCliques']= 7516
    BaseLogPartFctRef[ 439 ]['NbSites']= 2944
    BaseLogPartFctRef[ 439 ]['StdNgbhDivMoyNgbh']= 0.187317659209
    BaseLogPartFctRef[440 ]={}
    BaseLogPartFctRef[440]['LogPF']=np.array([3276.1,3407.1,3541.9,3679.2,3822.7,3972.1,4130.9,4300.6,4481.3,4682.1,4920.1,5198.0,5500.4,5825.0,6159.3,6505.0,6857.4,7214.1,7572.6,7938.2,8309.2,8683.1,9057.7,9432.1,9809.5,10187.0,10565.6,10945.0,11324.8,11704.9])
    BaseLogPartFctRef[ 440 ]['NbLabels']= 3
    BaseLogPartFctRef[ 440 ]['NbCliques']= 7649
    BaseLogPartFctRef[ 440 ]['NbSites']= 2982
    BaseLogPartFctRef[ 440 ]['StdNgbhDivMoyNgbh']= 0.178045474733
    BaseLogPartFctRef[441 ]={}
    BaseLogPartFctRef[441]['LogPF']=np.array([3202.5,3325.7,3452.9,3586.0,3723.3,3866.2,4013.7,4180.0,4350.2,4542.0,4768.4,5017.3,5297.2,5604.2,5918.9,6250.4,6584.7,6930.0,7277.4,7629.9,7983.4,8339.0,8699.3,9059.3,9421.0,9782.5,10146.6,10511.9,10876.1,11241.6])
    BaseLogPartFctRef[ 441 ]['NbLabels']= 3
    BaseLogPartFctRef[ 441 ]['NbCliques']= 7359
    BaseLogPartFctRef[ 441 ]['NbSites']= 2915
    BaseLogPartFctRef[ 441 ]['StdNgbhDivMoyNgbh']= 0.193835579538
    BaseLogPartFctRef[442 ]={}
    BaseLogPartFctRef[442]['LogPF']=np.array([3248.6,3377.4,3507.9,3644.8,3787.5,3932.5,4083.0,4247.3,4422.7,4637.9,4872.3,5143.2,5441.7,5754.0,6079.6,6421.7,6766.7,7123.1,7485.2,7846.0,8213.2,8580.5,8952.3,9324.4,9697.2,10071.3,10445.2,10820.9,11196.7,11573.7])
    BaseLogPartFctRef[ 442 ]['NbLabels']= 3
    BaseLogPartFctRef[ 442 ]['NbCliques']= 7576
    BaseLogPartFctRef[ 442 ]['NbSites']= 2957
    BaseLogPartFctRef[ 442 ]['StdNgbhDivMoyNgbh']= 0.184386902331
    BaseLogPartFctRef[443 ]={}
    BaseLogPartFctRef[443]['LogPF']=np.array([3197.0,3321.4,3452.2,3586.4,3723.9,3869.1,4020.5,4178.0,4350.9,4550.1,4779.6,5037.7,5323.7,5637.9,5962.0,6296.9,6632.2,6981.5,7332.1,7686.7,8044.1,8404.5,8767.6,9130.0,9495.5,9860.6,10226.0,10593.4,10960.6,11328.4])
    BaseLogPartFctRef[ 443 ]['NbLabels']= 3
    BaseLogPartFctRef[ 443 ]['NbCliques']= 7402
    BaseLogPartFctRef[ 443 ]['NbSites']= 2910
    BaseLogPartFctRef[ 443 ]['StdNgbhDivMoyNgbh']= 0.185876474335
    BaseLogPartFctRef[444 ]={}
    BaseLogPartFctRef[444]['LogPF']=np.array([3218.9,3348.0,3479.1,3614.7,3754.2,3900.8,4053.5,4214.5,4389.3,4592.9,4823.4,5085.7,5374.7,5676.4,6004.2,6341.0,6682.7,7032.1,7386.5,7746.8,8107.7,8472.3,8839.3,9207.4,9576.5,9946.5,10317.3,10687.8,11059.5,11432.2])
    BaseLogPartFctRef[ 444 ]['NbLabels']= 3
    BaseLogPartFctRef[ 444 ]['NbCliques']= 7491
    BaseLogPartFctRef[ 444 ]['NbSites']= 2930
    BaseLogPartFctRef[ 444 ]['StdNgbhDivMoyNgbh']= 0.188793217594
    BaseLogPartFctRef[445 ]={}
    BaseLogPartFctRef[445]['LogPF']=np.array([3276.1,3405.3,3539.7,3679.7,3822.3,3971.0,4126.7,4288.0,4475.9,4677.3,4922.6,5209.6,5505.6,5816.0,6149.9,6498.9,6854.2,7212.8,7578.4,7945.7,8316.7,8691.4,9068.0,9446.2,9825.6,10205.6,10586.8,10969.3,11351.6,11734.7])
    BaseLogPartFctRef[ 445 ]['NbLabels']= 3
    BaseLogPartFctRef[ 445 ]['NbCliques']= 7693
    BaseLogPartFctRef[ 445 ]['NbSites']= 2982
    BaseLogPartFctRef[ 445 ]['StdNgbhDivMoyNgbh']= 0.180774156288
    BaseLogPartFctRef[446 ]={}
    BaseLogPartFctRef[446]['LogPF']=np.array([3201.4,3327.5,3456.8,3590.9,3729.2,3875.2,4025.3,4190.0,4367.6,4559.8,4793.6,5058.7,5340.1,5647.4,5973.2,6309.0,6649.2,6997.9,7350.3,7703.5,8063.1,8422.9,8786.4,9149.0,9514.0,9881.2,10248.7,10617.1,10986.4,11355.6])
    BaseLogPartFctRef[ 446 ]['NbLabels']= 3
    BaseLogPartFctRef[ 446 ]['NbCliques']= 7436
    BaseLogPartFctRef[ 446 ]['NbSites']= 2914
    BaseLogPartFctRef[ 446 ]['StdNgbhDivMoyNgbh']= 0.188007135336
    BaseLogPartFctRef[447 ]={}
    BaseLogPartFctRef[447]['LogPF']=np.array([3194.8,3320.0,3448.9,3582.9,3723.9,3868.6,4022.5,4182.7,4357.8,4549.8,4760.5,5023.1,5324.8,5641.1,5965.8,6302.4,6645.5,6990.8,7343.7,7702.0,8058.5,8421.5,8784.8,9150.9,9518.0,9885.3,10253.6,10623.0,10992.4,11362.6])
    BaseLogPartFctRef[ 447 ]['NbLabels']= 3
    BaseLogPartFctRef[ 447 ]['NbCliques']= 7448
    BaseLogPartFctRef[ 447 ]['NbSites']= 2908
    BaseLogPartFctRef[ 447 ]['StdNgbhDivMoyNgbh']= 0.190136547931
    BaseLogPartFctRef[448 ]={}
    BaseLogPartFctRef[448]['LogPF']=np.array([3268.4,3396.3,3530.5,3667.4,3808.6,3954.7,4108.8,4269.6,4446.9,4639.4,4875.9,5147.9,5447.2,5764.9,6092.5,6431.7,6781.7,7142.8,7503.9,7867.8,8234.6,8606.3,8979.3,9352.1,9727.5,10104.4,10481.8,10860.0,11238.3,11616.9])
    BaseLogPartFctRef[ 448 ]['NbLabels']= 3
    BaseLogPartFctRef[ 448 ]['NbCliques']= 7624
    BaseLogPartFctRef[ 448 ]['NbSites']= 2975
    BaseLogPartFctRef[ 448 ]['StdNgbhDivMoyNgbh']= 0.187188178033
    BaseLogPartFctRef[449 ]={}
    BaseLogPartFctRef[449]['LogPF']=np.array([3284.9,3414.7,3548.6,3686.3,3829.8,3978.2,4134.3,4302.9,4485.8,4687.8,4922.4,5192.6,5486.0,5790.9,6118.6,6462.1,6814.8,7174.7,7539.7,7908.1,8278.6,8652.0,9026.0,9404.1,9782.3,10162.6,10543.6,10923.5,11304.6,11685.6])
    BaseLogPartFctRef[ 449 ]['NbLabels']= 3
    BaseLogPartFctRef[ 449 ]['NbCliques']= 7681
    BaseLogPartFctRef[ 449 ]['NbSites']= 2990
    BaseLogPartFctRef[ 449 ]['StdNgbhDivMoyNgbh']= 0.179728652519
    BaseLogPartFctRef[450 ]={}
    BaseLogPartFctRef[450]['LogPF']=np.array([3189.3,3313.5,3444.1,3576.7,3713.4,3855.6,4006.1,4168.8,4340.7,4528.2,4758.5,5022.5,5308.8,5619.4,5940.4,6267.4,6605.6,6951.5,7298.7,7653.0,8009.4,8370.8,8732.5,9096.7,9459.4,9824.6,10190.9,10556.6,10924.3,11292.2])
    BaseLogPartFctRef[ 450 ]['NbLabels']= 3
    BaseLogPartFctRef[ 450 ]['NbCliques']= 7399
    BaseLogPartFctRef[ 450 ]['NbSites']= 2903
    BaseLogPartFctRef[ 450 ]['StdNgbhDivMoyNgbh']= 0.19326362073
    BaseLogPartFctRef[451 ]={}
    BaseLogPartFctRef[451]['LogPF']=np.array([3247.5,3376.0,3507.8,3643.6,3786.9,3932.4,4089.3,4254.6,4434.9,4633.5,4859.4,5131.3,5435.8,5749.9,6080.5,6428.3,6777.0,7130.5,7489.8,7850.9,8216.3,8586.0,8955.8,9329.3,9703.7,10078.8,10453.8,10829.5,11206.6,11583.9])
    BaseLogPartFctRef[ 451 ]['NbLabels']= 3
    BaseLogPartFctRef[ 451 ]['NbCliques']= 7590
    BaseLogPartFctRef[ 451 ]['NbSites']= 2956
    BaseLogPartFctRef[ 451 ]['StdNgbhDivMoyNgbh']= 0.184517318445
    BaseLogPartFctRef[452 ]={}
    BaseLogPartFctRef[452]['LogPF']=np.array([3228.8,3356.3,3485.7,3621.0,3761.3,3909.7,4059.5,4221.9,4396.6,4595.5,4822.8,5088.9,5379.4,5698.5,6030.3,6363.4,6709.9,7060.6,7419.8,7782.6,8142.1,8507.5,8875.6,9243.1,9612.2,9983.1,10354.9,10726.1,11099.1,11471.8])
    BaseLogPartFctRef[ 452 ]['NbLabels']= 3
    BaseLogPartFctRef[ 452 ]['NbCliques']= 7500
    BaseLogPartFctRef[ 452 ]['NbSites']= 2939
    BaseLogPartFctRef[ 452 ]['StdNgbhDivMoyNgbh']= 0.182635741583
    BaseLogPartFctRef[453 ]={}
    BaseLogPartFctRef[453]['LogPF']=np.array([3284.9,3413.8,3547.5,3686.1,3829.4,3978.0,4133.6,4303.0,4487.5,4688.3,4919.4,5179.1,5473.2,5790.5,6125.7,6466.6,6821.5,7185.8,7553.7,7921.9,8291.8,8666.3,9040.8,9417.7,9795.6,10174.2,10554.7,10934.9,11316.7,11698.6])
    BaseLogPartFctRef[ 453 ]['NbLabels']= 3
    BaseLogPartFctRef[ 453 ]['NbCliques']= 7674
    BaseLogPartFctRef[ 453 ]['NbSites']= 2990
    BaseLogPartFctRef[ 453 ]['StdNgbhDivMoyNgbh']= 0.177543041584
    BaseLogPartFctRef[454 ]={}
    BaseLogPartFctRef[454]['LogPF']=np.array([3186.0,3310.2,3438.2,3571.8,3707.3,3850.0,3997.3,4155.2,4330.9,4538.3,4765.1,5017.3,5298.8,5598.4,5913.2,6240.5,6578.6,6925.8,7271.4,7620.9,7975.4,8330.6,8687.6,9048.2,9408.4,9770.0,10132.7,10495.8,10859.6,11224.0])
    BaseLogPartFctRef[ 454 ]['NbLabels']= 3
    BaseLogPartFctRef[ 454 ]['NbCliques']= 7337
    BaseLogPartFctRef[ 454 ]['NbSites']= 2900
    BaseLogPartFctRef[ 454 ]['StdNgbhDivMoyNgbh']= 0.188337653603
    BaseLogPartFctRef[455 ]={}
    BaseLogPartFctRef[455]['LogPF']=np.array([3248.6,3373.7,3506.3,3643.3,3784.2,3931.3,4085.1,4249.7,4427.4,4617.6,4853.3,5132.5,5425.4,5740.6,6069.6,6406.2,6752.3,7109.4,7463.9,7822.5,8187.2,8553.2,8919.9,9290.3,9661.8,10033.2,10406.1,10780.3,11154.6,11528.9])
    BaseLogPartFctRef[ 455 ]['NbLabels']= 3
    BaseLogPartFctRef[ 455 ]['NbCliques']= 7534
    BaseLogPartFctRef[ 455 ]['NbSites']= 2957
    BaseLogPartFctRef[ 455 ]['StdNgbhDivMoyNgbh']= 0.181601409976
    BaseLogPartFctRef[456 ]={}
    BaseLogPartFctRef[456]['LogPF']=np.array([3193.7,3318.8,3447.5,3579.9,3717.8,3863.1,4009.1,4167.0,4343.3,4533.5,4760.0,5007.1,5299.8,5606.9,5919.3,6242.7,6583.8,6928.4,7276.8,7633.5,7991.1,8350.2,8710.9,9072.6,9435.8,9801.2,10167.6,10533.9,10900.9,11269.2])
    BaseLogPartFctRef[ 456 ]['NbLabels']= 3
    BaseLogPartFctRef[ 456 ]['NbCliques']= 7394
    BaseLogPartFctRef[ 456 ]['NbSites']= 2907
    BaseLogPartFctRef[ 456 ]['StdNgbhDivMoyNgbh']= 0.18430088024
    BaseLogPartFctRef[457 ]={}
    BaseLogPartFctRef[457]['LogPF']=np.array([3267.3,3397.3,3530.6,3670.7,3812.5,3964.1,4122.8,4285.3,4470.0,4670.4,4906.2,5182.3,5485.5,5803.4,6132.0,6478.7,6827.4,7187.7,7550.2,7921.7,8292.4,8661.7,9035.2,9409.8,9787.1,10164.6,10542.1,10920.6,11300.4,11680.3])
    BaseLogPartFctRef[ 457 ]['NbLabels']= 3
    BaseLogPartFctRef[ 457 ]['NbCliques']= 7639
    BaseLogPartFctRef[ 457 ]['NbSites']= 2974
    BaseLogPartFctRef[ 457 ]['StdNgbhDivMoyNgbh']= 0.181737755074
    BaseLogPartFctRef[458 ]={}
    BaseLogPartFctRef[458]['LogPF']=np.array([3206.8,3332.3,3462.2,3598.1,3737.9,3881.7,4031.3,4191.1,4375.1,4570.4,4805.7,5066.9,5351.0,5662.8,5990.8,6323.3,6663.8,7013.8,7363.3,7722.3,8082.5,8442.8,8809.3,9175.4,9543.2,9912.0,10280.9,10651.4,11022.2,11392.9])
    BaseLogPartFctRef[ 458 ]['NbLabels']= 3
    BaseLogPartFctRef[ 458 ]['NbCliques']= 7468
    BaseLogPartFctRef[ 458 ]['NbSites']= 2919
    BaseLogPartFctRef[ 458 ]['StdNgbhDivMoyNgbh']= 0.182857692166
    BaseLogPartFctRef[459 ]={}
    BaseLogPartFctRef[459]['LogPF']=np.array([3251.9,3380.2,3512.2,3650.0,3794.5,3943.3,4102.2,4267.4,4450.3,4656.9,4888.6,5160.7,5453.6,5767.8,6107.2,6449.7,6802.8,7160.3,7525.8,7892.2,8261.2,8631.8,9006.4,9381.9,9760.1,10137.9,10516.0,10895.5,11275.2,11655.8])
    BaseLogPartFctRef[ 459 ]['NbLabels']= 3
    BaseLogPartFctRef[ 459 ]['NbCliques']= 7644
    BaseLogPartFctRef[ 459 ]['NbSites']= 2960
    BaseLogPartFctRef[ 459 ]['StdNgbhDivMoyNgbh']= 0.18019317349
    BaseLogPartFctRef[460 ]={}
    BaseLogPartFctRef[460]['LogPF']=np.array([3298.0,3429.9,3565.3,3704.0,3850.9,4000.8,4159.4,4324.6,4506.0,4712.3,4948.8,5228.2,5528.9,5863.0,6205.0,6558.1,6911.7,7274.6,7643.9,8016.0,8391.0,8768.7,9150.1,9532.8,9916.1,10299.9,10684.6,11069.3,11455.1,11841.2])
    BaseLogPartFctRef[ 460 ]['NbLabels']= 3
    BaseLogPartFctRef[ 460 ]['NbCliques']= 7761
    BaseLogPartFctRef[ 460 ]['NbSites']= 3002
    BaseLogPartFctRef[ 460 ]['StdNgbhDivMoyNgbh']= 0.175609354911
    BaseLogPartFctRef[461 ]={}
    BaseLogPartFctRef[461]['LogPF']=np.array([3256.3,3383.5,3516.4,3653.7,3795.6,3943.7,4099.6,4263.6,4439.1,4638.1,4885.2,5152.7,5439.5,5758.8,6092.7,6437.8,6785.5,7146.4,7501.6,7865.9,8232.9,8601.0,8970.8,9344.1,9718.3,10093.2,10469.5,10845.5,11221.9,11599.4])
    BaseLogPartFctRef[ 461 ]['NbLabels']= 3
    BaseLogPartFctRef[ 461 ]['NbCliques']= 7590
    BaseLogPartFctRef[ 461 ]['NbSites']= 2964
    BaseLogPartFctRef[ 461 ]['StdNgbhDivMoyNgbh']= 0.179845616885
    BaseLogPartFctRef[462 ]={}
    BaseLogPartFctRef[462]['LogPF']=np.array([3172.8,3294.7,3422.6,3553.8,3691.1,3830.1,3978.7,4137.5,4305.3,4492.3,4721.5,4975.7,5257.2,5551.4,5861.3,6181.5,6517.3,6859.0,7200.4,7545.4,7894.4,8245.6,8601.4,8958.2,9316.9,9674.2,10033.6,10393.7,10754.7,11116.5])
    BaseLogPartFctRef[ 462 ]['NbLabels']= 3
    BaseLogPartFctRef[ 462 ]['NbCliques']= 7272
    BaseLogPartFctRef[ 462 ]['NbSites']= 2888
    BaseLogPartFctRef[ 462 ]['StdNgbhDivMoyNgbh']= 0.192011604121
    BaseLogPartFctRef[463 ]={}
    BaseLogPartFctRef[463]['LogPF']=np.array([3245.3,3371.0,3502.2,3638.7,3780.3,3927.4,4079.9,4247.1,4427.2,4628.6,4868.6,5137.0,5439.6,5749.4,6073.5,6407.1,6752.9,7106.4,7464.9,7827.4,8192.5,8559.9,8930.8,9301.3,9672.3,10045.1,10418.6,10792.9,11168.0,11543.7])
    BaseLogPartFctRef[ 463 ]['NbLabels']= 3
    BaseLogPartFctRef[ 463 ]['NbCliques']= 7558
    BaseLogPartFctRef[ 463 ]['NbSites']= 2954
    BaseLogPartFctRef[ 463 ]['StdNgbhDivMoyNgbh']= 0.188403104636
    BaseLogPartFctRef[464 ]={}
    BaseLogPartFctRef[464]['LogPF']=np.array([3224.4,3351.2,3482.4,3617.1,3758.1,3904.4,4055.5,4217.8,4394.6,4599.1,4831.0,5104.9,5405.5,5722.0,6053.2,6389.7,6734.4,7087.1,7446.2,7809.8,8172.7,8539.6,8906.5,9274.7,9645.1,10017.0,10389.3,10763.0,11137.1,11511.9])
    BaseLogPartFctRef[ 464 ]['NbLabels']= 3
    BaseLogPartFctRef[ 464 ]['NbCliques']= 7535
    BaseLogPartFctRef[ 464 ]['NbSites']= 2935
    BaseLogPartFctRef[ 464 ]['StdNgbhDivMoyNgbh']= 0.183660633269


    V_Beta_Ref=np.array([0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45])

    return BaseLogPartFctRef,V_Beta_Ref


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Cpt_Vec_Estim_lnZ_Graph_fast(RefGraph,LabelsNb,MaxErrorAllowed=5,BetaMax=1.4,BetaStep=0.05):
    """
    Estimate ln(Z(beta)) of Potts fields. The default Beta grid is between 0. and 1.4 with
    a step of 0.05. Extrapolation algorithm is used. Fast estimates are only performed for
    Ising fields (2 labels). Reference partition functions were pre-computed on Ising fields
    designed on regular and non-regular grids. They all respect a 6-connectivity system.
    input:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
            and contains the list of its neighbors entry location in the graph.
            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.  => There exists i such that RefGraph[10][i]=2
        * LabelsNb: possible number of labels in each site of the graph
        * MaxErrorAllowed: maximum error allowed in the graph estimation (in percents).
        * BetaMax: Z(beta,mask) will be computed for beta between 0 and BetaMax. Maximum considered value is 1.4
        * BetaStep: gap between two considered values of beta. Actual gaps are not exactly those asked but very close.
    output:
        * Est_lnZ: Vector containing the ln(Z(beta)) estimates
        * V_Beta: Vector of the same size as VecExpectZ containing the corresponding beta value
    """

    #launch a more general algorithm if the inputs are not appropriate
    if (LabelsNb!=2 and LabelsNb!=3) or BetaMax>1.4:
        [Est_lnZ,V_Beta]=Cpt_Vec_Estim_lnZ_Graph(RefGraph,LabelsNb,SamplesNb=30,BetaMax=BetaMax,BetaStep=BetaStep,GraphWeight=None)
        return Est_lnZ,V_Beta

    #initialisation

    #...default returned values
    V_Beta=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4])
    Est_lnZ=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    #...load reference partition functions
    [BaseLogPartFctRef,V_Beta_Ref]=LoadBaseLogPartFctRef()

    NbSites=[len(RefGraph)]
    NbCliques=[sum( [len(nl) for nl in RefGraph] ) ]
    #...NbSites
    s=len(RefGraph)
    #...NbCliques
    NbCliquesTmp=0
    for j in xrange(len(RefGraph)):
        NbCliquesTmp=NbCliquesTmp+len(RefGraph[j])
    c=NbCliquesTmp/2
    NbCliques.append(c)
    #...StdVal Nb neighbors / Moy Nb neighbors
    StdValCliquesPerSiteTmp=0.

    nc = NbCliques[-1] + 0.
    ns = NbSites[-1] + 0.
    for j in xrange(len(RefGraph)):
        StdValCliquesPerSiteTmp = StdValCliquesPerSiteTmp \
            + ( (nc/ns-len(RefGraph[j])/2.)**2. ) / ns
        StdNgbhDivMoyNgbh = np.sqrt(StdValCliquesPerSiteTmp) \
            / ( nc/(ns-1.) )

    #extrapolation algorithm
    Best_MaxError=10000000.

    for i in BaseLogPartFctRef.keys():
        if BaseLogPartFctRef[i]['NbLabels']==LabelsNb:
            MaxError=np.abs((BaseLogPartFctRef[i]['NbSites']-1.)*((1.*c)/(1.*BaseLogPartFctRef[i]['NbCliques']))-(s-1.))*np.log(LabelsNb*1.)  #error at beta=0
            MaxError=MaxError+(np.abs(BaseLogPartFctRef[i]['StdNgbhDivMoyNgbh']-StdNgbhDivMoyNgbh))  #penalty added to the error at zero to penalyze different homogeneities of the neighboroud  (a bareer function would be cleaner for the conversion in percents)
            MaxError=MaxError*100./(s*np.log(LabelsNb*1.))    #to have a percentage of error
            if MaxError<Best_MaxError:
                Best_MaxError=MaxError
                BestI=i

    if Best_MaxError<MaxErrorAllowed:
        Est_lnZ=((c*1.)/(BaseLogPartFctRef[BestI]['NbCliques']*1.))*BaseLogPartFctRef[BestI]['LogPF']+(1-(c*1.)/(BaseLogPartFctRef[BestI]['NbCliques']*1.))*np.log(LabelsNb*1.)
        V_Beta=V_Beta_Ref.copy()
    else:
        #print 'launch an adapted function'
        [Est_lnZ,V_Beta] = Cpt_Vec_Estim_lnZ_Graph(RefGraph,LabelsNb,SamplesNb=30,BetaMax=BetaMax,BetaStep=BetaStep,GraphWeight=None)

    #print 'V_Beta be4 resampling:', V_Beta
    #reduction of the domain
    if (BetaMax<1.4):
        temp=0
        while V_Beta[temp]<BetaMax and temp<V_Beta.shape[0]-2:
            temp=temp+1
        V_Beta=V_Beta[:temp]
        Est_lnZ=Est_lnZ[:temp]

    #domain resampling
    resamplingMethod = 'ply'
    if (abs(BetaStep-0.05)>0.0001):
        if resamplingMethod == 'linear':
            v_Beta_Resample = []
            cpt=0.
            while cpt<BetaMax+0.0001:
                v_Beta_Resample.append(cpt)
                cpt=cpt+BetaStep
            Est_lnZ = resampleToGrid(np.array(V_Beta),np.array(Est_lnZ),
                                     np.array(v_Beta_Resample))
            V_Beta = v_Beta_Resample
        elif resamplingMethod == 'ply':
            interpolator = scipy.interpolate.interp1d(V_Beta, Est_lnZ,
                                                      kind='cubic')
            #print 'V_Beta[-1]+BetaStep:', V_Beta[-1],'+',BetaStep
            targetBeta = np.arange(0, BetaMax + BetaStep, BetaStep)
            Est_lnZ = interpolator(targetBeta)
            #Est_lnZ = scipy.interpolate.spline(V_Beta, Est_lnZ, targetBeta,
            #                                   order=3)
            #Est_lnZ = scipy.interpolate.krogh_interpolate(V_Beta, Est_lnZ,
            #                                              targetBeta)
            #print 'Est_lnZ:', Est_lnZ
            #print 'targetBeta:', targetBeta
            V_Beta = targetBeta
        else:
            raise Exception('Unknown resampling method: %s' %resamplingMethod)
    return Est_lnZ,V_Beta



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Cpt_Vec_Estim_lnZ_Graph_fast2(RefGraph,BetaMax=1.4,BetaStep=0.05):
    """
    Estimate ln(Z(beta)) of Ising fields (2 labels). The default Beta grid is between 0. and 1.4 with
    a step of 0.05. Bilinar estimation with the number of sites and cliques is used. The bilinear functions
    were estimated using bilinear regression on reference partition functions on 240 non-regular grids and with
    respect to a 6-connectivity system. (Pfs are found in LoadBaseLogPartFctRef -> PFs 0:239)

    input:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
            and contains the list of its neighbors entry location in the graph.
            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.  => There exists i such that RefGraph[10][i]=2
        * BetaMax: Z(beta,mask) will be computed for beta between 0 and BetaMax. Maximum considered value is 1.4
        * BetaStep: gap between two considered values of beta. Actual gaps are not exactly those asked but very close.
    output:
        * Est_lnZ: Vector containing the ln(Z(beta)) estimates
        * V_Beta: Vector of the same size as VecExpectZ containing the corresponding beta value
    """

    #launch a more general algorithm if the inputs are not appropriate
    if LabelsNb!=2 or BetaMax>1.4:
        [Est_lnZ,V_Beta]=Cpt_Vec_Estim_lnZ_Graph(RefGraph,LabelsNb,SamplesNb=20,BetaMax=BetaMax,BetaStep=BetaStep,GraphWeight=None)
        return Est_lnZ,V_Beta

    #initialisation

    #...default returned values
    V_Beta=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4])
    Est_lnZ=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    #...NbSites
    s=len(RefGraph)
    #...NbCliques
    NbCliquesTmp=0
    for j in xrange(len(RefGraph)):
        NbCliquesTmp=NbCliquesTmp+len(RefGraph[j])
    c=NbCliquesTmp/2


    #extrapolation algorithm
    Est_lnZ[0]= 0.051 * c + 0.693  * s  -  0.0004
    Est_lnZ[1]= 0.105 * c + 0.692 *  s  +  0.003
    Est_lnZ[2]= 0.162 * c + 0.691 *  s  + 0.012
    Est_lnZ[3]= 0.224 * c + 0.686 *  s  + 0.058
    Est_lnZ[4]= 0.298 * c + 0.663 *  s  + 0.314
    Est_lnZ[5]= 0.406 * c + 0.580 *  s  + 1.26
    Est_lnZ[6]= 0.537 * c + 0.467 *  s  + 2.34
    Est_lnZ[7]= 0.669 * c + 0.363 *  s  + 3.07
    Est_lnZ[8]= 0.797 * c + 0.281 *  s  + 3.39
    Est_lnZ[9]= 0.919 * c + 0.219 *  s  + 3.41
    Est_lnZ[10]=1.035 * c + 0.173 *  s  +  3.28
    Est_lnZ[11]=1.148 * c + 0.137 *  s  + 3.08
    Est_lnZ[12]=1.258 * c + 0.110 *  s  +  2.87
    Est_lnZ[13]=1.366 * c + 0.089 *  s  + 2.66

    #reduction of the domain
    if (BetaMax<1.4):
        temp=0
        while V_Beta[temp]<BetaMax and temp<V_Beta.shape[0]-2:
            temp=temp+1
        V_Beta=V_Beta[:temp]
        Est_lnZ=Est_lnZ[:temp]

    #domain resampling
    if (abs(BetaStep-0.05)>0.0001):
        v_Beta_Resample=[]
        cpt=0.
        while cpt<BetaMax+0.0001:
            v_Beta_Resample.append(cpt)
            cpt=cpt+BetaStep
        Est_lnZ=resampleToGrid(np.array(V_Beta),np.array(Est_lnZ),np.array(v_Beta_Resample))
        V_Beta=v_Beta_Resample

    return Est_lnZ,V_Beta


def logpf_ising_onsager(size, beta):
    """
    Calculate log partition function in terms of beta for an Ising field
    of size 'size'. 'beta' can be scalar or numpy.array.
    Assumptions: the field is 2D, squared, toroidal and has 4-connectivity
    """
    coshb = np.cosh(beta)
    u = 2 * np.sinh(beta) / (coshb*coshb)
    if np.isscalar(beta):
        psi = np.zeros(1)
        u = [u]
    else:
        psi = np.zeros(len(beta))
    for iu,vu in enumerate(u):
            x = np.arange(0, np.pi, 0.01)
            sinx = np.sin(x)
            y = np.log( (1 + (1 - vu*vu*sinx*sinx)**.5)/2 )
            psi[iu] = 1/(2*np.pi) * np.trapz(y,x)
    if np.isscalar(beta):
        return size * (beta + np.log(2*coshb + psi[0]))
    else:
        return size * (beta + np.log(2*coshb + psi))






#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Estim_lnZ_Onsager(n,beta):
    """
    Estimate ln(Z(beta)) using Onsager technique (2D periodic fields - 2 labels - 4 connectivity)
    input:
        * n: number of sites
        * beta: beta
    output:
        * LogZ: ln(Z(beta)) estimate
    """
    #estimate u(beta)
    u=2.*scipy.sinh(beta)/(scipy.cosh(beta)*scipy.cosh(beta))

    #estimate psi(u(beta))
    NbSteps=1000
    DeltaX=scipy.pi/NbSteps
    integra=0.
    for i in xrange(NbSteps):
        x=scipy.pi*(i+0.5)/NbSteps
        integra+=(scipy.log((1.+scipy.sqrt(1-u*u*scipy.sin(x)*scipy.sin(x)))/2.))*DeltaX
    Psi=integra/(2.*scipy.pi)

    #estimate Log(Z)
    LogZ=n*(beta+scipy.log(2.*scipy.cosh(beta))+Psi)


    return LogZ

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Cpt_Vec_Estim_lnZ_Onsager(n,BetaMax=1.2,BetaStep=0.05):
    """
    Estimate ln(Z(beta)) Onsager using Onsager technique (2D periodic fields - 2 labels - 4 connectivity)
    input:
        * n: number of sites
        * BetaMax: Z(beta,mask) will be computed for beta between 0 and BetaMax. Maximum considered value is 1.2.
        * BetaStep: gap between two considered values of beta. Actual gaps are not exactly those asked but very close.
    output:
        * Est_lnZ: Vector containing the ln(Z(beta)) estimates
        * V_Beta: Vector of the same size as VecExpectZ containing the corresponding beta value
    """

    #initialization
    BetaLoc=0.
    ListBetaVal=[]
    ListLnZ=[]

    #compute the values of beta and lnZ
    while BetaLoc<BetaMax:
        LnZLoc=Estim_lnZ_Onsager(n,BetaLoc)
        ListLnZ.append(LnZLoc)
        ListBetaVal.append(BetaLoc)
        BetaLoc=BetaLoc+BetaStep

    #cast the result into an array
    Est_lnZ=_N.array(ListLnZ)
    V_Beta=_N.array(ListBetaVal)

    #return the estimated values
    return Est_lnZ,V_Beta





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Cpt_Vec_Estim_lnZ_OLD_Graph(RefGraph,LabelsNb,SamplesNb=50,BetaMax=1.0,BetaStep=0.01,GraphWeight=None):
    """
    Useless now!

    Estimates ln(Z) for fields of a given size and Beta values between 0 and BetaMax
    input:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
        * LabelsNb: number of labels
        * BetaMax: Z(beta,mask) will be computed for beta between 0 and BetaMax
        * BetaStep: gap between two considered values of bseta
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding
                       edge in RefGraph. If not defined the weights are set to 1.0.
    output:
        * VecEstim_lnZ: Vector containing the ln(Z(beta,mask)) estimates
        * VecBetaVal: Vector of the same size as VecExpectZ containing the corresponding beta value
    """

    #initialization

    BetaLoc=0
    ListExpectU=[]
    ListBetaVal=[]

    if GraphWeight==None:
        GraphWeight=CptDefaultGraphWeight(RefGraph)

    GraphNodesLabels=CptDefaultGraphNodesLabels(RefGraph)
    GraphLinks=CptDefaultGraphLinks(RefGraph)
    RefGrphNgbhPosi=CptRefGrphNgbhPosi(RefGraph)

    #compute all E(U|beta)...
    while BetaLoc<BetaMax:
        BetaLoc=BetaLoc+BetaStep
        ExpU_loc=Cpt_Expected_U_graph(RefGraph,BetaLoc,LabelsNb,SamplesNb,GraphWeight=GraphWeight,GraphNodesLabels=GraphNodesLabels,GraphLinks=GraphLinks,RefGrphNgbhPosi=RefGrphNgbhPosi)
        ListExpectU.append(ExpU_loc)
        ListBetaVal.append(BetaLoc)

        pyhrf.verbose(2, 'beta=%1.4f ->  exp(U)=%1.4f' \
                      %(BetaLoc,ExpU_loc))

    VecEstim_lnZ=np.zeros(len(ListExpectU))
    VecBetaVal=np.zeros(len(ListExpectU))

    for i in xrange(len(ListExpectU)):
        VecBetaVal[i]=ListBetaVal[i]
        if i==0:
            VecEstim_lnZ[i]=len(RefGraph)*np.log(LabelsNb)+ListExpectU[0]*BetaStep/2
        else:
            VecEstim_lnZ[i]=VecEstim_lnZ[i-1]+(ListExpectU[i-1]+ListExpectU[i])*BetaStep/2

    return VecEstim_lnZ,VecBetaVal

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Cpt_Exact_lnZ_graph(RefGraph,beta,LabelsNb,GraphWeight=None):
    """
    Computes the logarithm of the exact partition function Z(\beta).
    input:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
        * beta: spatial regularization parameter
        * LabelsNb: number of labels in each site (typically 2 or 3)
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding
                       edge in RefGraph. If not defined the weights are set to 1.0.
    output:
        * exact_lnZ: exact value of ln(Z)
    """

    #initialization
    if GraphWeight==None:
        GraphWeight=CptDefaultGraphWeight(RefGraph)

    GraphNodesLabels=CptDefaultGraphNodesLabels(RefGraph)

    VoxelsNb=len(RefGraph)

    #sum of the energies U for all possible configurations of GraphNodesLabels
    Config_ID_max=LabelsNb**VoxelsNb

    Z_exact=0
    if LabelsNb==2: #use of np.binary_repr instead of np.base_repr because it is far faster but only designed for base two numbers
        for Config_ID in xrange(Config_ID_max):
            if Config_ID==0:  #handle a problem with 'binary_repr' which write binary_repr(0,VoxelsNb)='0'
                for i in xrange(VoxelsNb):
                    GraphNodesLabels[i]=0
            else:
                for i in xrange(VoxelsNb):
                    GraphNodesLabels[i]=int(np.binary_repr(Config_ID,VoxelsNb)[i])
            #print GraphNodesLabels
            Z_exact=Z_exact+np.exp(beta*Cpt_U_graph(RefGraph,GraphNodesLabels,GraphWeight=GraphWeight))
    else:
        for Config_ID in xrange(Config_ID_max):
            for i in xrange(VoxelsNb):
                GraphNodesLabels[i]=int(np.base_repr(Config_ID,base=LabelsNb,padding=VoxelsNb)[-1-i])
            #print GraphNodesLabels
            Z_exact=Z_exact+np.exp(beta*Cpt_U_graph(RefGraph,GraphNodesLabels,GraphWeight=GraphWeight))
    exact_lnZ=np.log(Z_exact)

    return exact_lnZ


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Cpt_Distrib_P_beta_graph(RefGraph,GraphNodesLabels,VecEstim_lnZ,VecBetaVal,
                             thresh=1.5,GraphWeight=None):

    """
    Computes the distribution P(beta|q)
    input:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
        * GraphNodesLabels: Nodes labels. GraphNodesLabels[i] is the node i label.
        * VecEstim_lnZ: Vector containing the ln(Z(beta,mask)) estimates (in accordance with the defined graph).
        * VecBetaVal: Vector of the same size as VecExpectZ containing the corresponding beta
                      value (in accordance with the defined graph).
        * thresh: the prior on beta is uniform between 0 and thresh and linearly decrease between thresh and VecBetaVal[-1]
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding
                       edge in RefGraph. If not defined the weights are set to 1.0.
    output:
        * Vec_P_Beta: contains the P(beta|q) values  (consistant with VecBetaVal).
    """

    #initialization
    if GraphWeight==None:
        GraphWeight=CptDefaultGraphWeight(RefGraph)

    BetaStep=VecBetaVal[1]-VecBetaVal[0]
    BetaLoc=0

    Vec_P_Beta=VecEstim_lnZ*0.0

    #computes all P(beta_i|q_i)
    #cpt the Energy
    Energy=Cpt_U_graph(RefGraph,GraphNodesLabels,GraphWeight=GraphWeight)
    #print 'Energy:', Energy

    #prior normalization
    if thresh>VecBetaVal[-1]:
        thresh=VecBetaVal[-1]

    PriorDenomi=(thresh)+((VecBetaVal[-1]-thresh)/2.)
    #print "PriorDenomi:", PriorDenomi

    for i in xrange(VecEstim_lnZ.shape[0]):
        #print 'VecBetaVal:', VecBetaVal[i]
        #print thresh
        if VecBetaVal[i]<thresh:
            P_beta=1.
        else:
            P_beta=(VecBetaVal[-1]-VecBetaVal[i])/(VecBetaVal[-1]-thresh)
        #print 'P_beta:', P_beta
        log_P_Beta=-VecEstim_lnZ[i]+Energy*VecBetaVal[i]+np.log(P_beta/PriorDenomi)
        #print 'log_P_Beta:', log_P_Beta
        Vec_P_Beta[i]=np.exp(log_P_Beta.astype(float_hires))
        if np.isnan(Vec_P_Beta[i]):
            Vec_P_Beta[i] = 0.
        #print 'Vec_P_Beta[i]', Vec_P_Beta[i]

    #print 'BetaStep:', BetaStep
    #print 'Vec_P_Beta:', Vec_P_Beta, Vec_P_Beta.sum()

    #print '(BetaStep*Vec_P_Beta.sum())', (BetaStep*Vec_P_Beta.sum())

    #Vec_P_Beta normalization
    Vec_P_Beta=Vec_P_Beta/(BetaStep*Vec_P_Beta.sum())
    return Vec_P_Beta

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Cpt_AcceptNewBeta_Graph(RefGraph,GraphNodesLabels,VecEstim_lnZ,VecBetaVal,CurrentBeta,sigma,thresh=1.2,GraphWeight=None):
    """
    Starting from a given Beta vector (1 value for each condition) 'CurrentBeta', computes new Beta values in 'NewBeta'
    using a Metropolis-Hastings step.
    input:
        * RefGraph: List which contains the connectivity graph. Each entry represents a node of the graph
                            and contains the list of its neighbors entry location in the graph.
                            ex: RefGraph[2][3]=10 means 3rd neighbour of the 2nd node is the 10th node.
                                 => There exists i such that RefGraph[10][i]=2
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding
                       edge in RefGraph. If not defined the weights are set to 1.0.
        * GraphNodesLabels: Nodes labels. GraphNodesLabels[i] is the node i label.
        * VecEstim_lnZ: Vector containing the ln(Z(beta,mask)) estimates (in accordance with the defined mask).
        * VecBetaVal: Vector of the same size as VecExpectZ containing the corresponding beta
                      value (in accordance with the defined mask).
        * CurrentBeta: Beta at the current iteration
        * sigma: such as NewBeta = CurrentBeta + N(0,sigma)
        * thresh: the prior on beta is uniform between 0 and thresh and linearly decrease between thresh and VecBetaVal[-1]
        * GraphWeight: Same shape as RefGraph. Each entry is the weight of the corresponding
                       edge in RefGraph. If not defined the weights are set to 1.0.
    output:
        * NewBeta: Contains the accepted beta value at the next iteration
    """


    #1) initialization
    betaMax = VecBetaVal[-1]
    betaMin = VecBetaVal[0]
    priorBetaMax = betaMax
    BetaStep=VecBetaVal[1]-VecBetaVal[0]

    if thresh>betaMax or thresh<0:
        thresh = betaMax

    if GraphWeight==None:
        GraphWeight=CptDefaultGraphWeight(RefGraph)

    bInf = (betaMin-CurrentBeta)/sigma
    bSup = (betaMax-CurrentBeta)/sigma
##     print 'betaMax :', betaMax, 'CurrentBeta :', CurrentBeta, 'sigma:', sigma
##     print 'cur beta :', CurrentBeta
##     print ' bInf =', bInf, 'bSup =', bSup
    u = truncRandn(1, a=bInf, b=bSup)
    #print ' u = ', u, 'sigma = ', sigma, '-> u*s=', u*sigma
    dBeta = sigma*u[0]
    NewBeta = CurrentBeta + dBeta
    #print '-> proposed beta:', NewBeta
    assert (NewBeta <= betaMax) and (NewBeta >= betaMin)

    #3.1) computes ln(Z(CurrentBeta|Mask)) estimation
    i=0
    while VecBetaVal[i]<CurrentBeta:
        i=i+1

    # First order interpolation of precomputed log-PF at currentBeta:
    ln_Z1_estim= VecEstim_lnZ[i-1] * (VecBetaVal[i]-CurrentBeta)/(VecBetaVal[i]-VecBetaVal[i-1]) \
        +VecEstim_lnZ[i] * (CurrentBeta-VecBetaVal[i-1])/(VecBetaVal[i]-VecBetaVal[i-1])

    #3.1.b) compute the prior P(CurrentBeta)    thresh=VecBetaVal[-1]

    #prior normalization
    if thresh>betaMax:
        thresh=betaMax

    PriorDenomi=(thresh)+((betaMax-thresh)/2.)

    if CurrentBeta<thresh:
        P_cur_beta=1.
    elif CurrentBeta<betaMax:
        P_cur_beta=(betaMax-CurrentBeta)/(betaMax-thresh)
    else:
        P_cur_beta=0.000001

    #3.2) computes ln(P(CurrentBeta|q,Mask)) estimation
    Energy=Cpt_U_graph(RefGraph,GraphNodesLabels,GraphWeight=GraphWeight)
    log_P_CurrentBeta = -ln_Z1_estim + Energy*CurrentBeta + np.log(P_cur_beta/PriorDenomi)


    #4.1) computes ln(Z(NewBeta|Mask)) estimation
    i=0
    while VecBetaVal[i]<NewBeta:
        i=i+1
    # First order interpolation of precomputed log-PF at  NewBeta:
    ln_Z2_estim=VecEstim_lnZ[i-1]*(VecBetaVal[i]-NewBeta)/(VecBetaVal[i]-VecBetaVal[i-1])+VecEstim_lnZ[i]*(NewBeta-VecBetaVal[i-1])/(VecBetaVal[i]-VecBetaVal[i-1])

    #4.1.b) compute the prior P(NewBeta)
    if NewBeta<thresh:
        P_new_beta=1.
    elif NewBeta<betaMax:
        P_new_beta=(betaMax-NewBeta)/(betaMax-thresh)
    else:
        P_new_beta=0.000001


    #4.2) computes ln(P(NewBeta|q,Mask)) estimation
    log_P_NewBeta=-ln_Z2_estim+Energy*NewBeta+np.log(P_new_beta/PriorDenomi)


    #5) compute A_NewBeta and accept or not the new beta value
    sigsqrt2 = sigma*2**.5
    log_g_new = erf((betaMax-NewBeta)/sigsqrt2) - erf((betaMin-NewBeta)/sigsqrt2)
    log_g_cur = erf((betaMax-CurrentBeta)/sigsqrt2) - erf((betaMin-CurrentBeta)/sigsqrt2)
    temp = np.exp(log_P_NewBeta - log_P_CurrentBeta + log_g_cur - log_g_new)
    A_NewBeta=min(1,temp)
    #print 'Accept ratio :', A_NewBeta
    if np.random.rand() > A_NewBeta:
        #print ' -> rejected !'
        NewBeta = CurrentBeta

    return NewBeta, dBeta, A_NewBeta



def beta_estim_obs_field(graph, labels, gridLnz, method='MAP',weights=None):
    """
    Estimate the amount of spatial correlation of an Ising observed field.
    'graph' is the neighbours list defining the topology
    'labels' is the field realisation
    'gridLnz' is the log-partition function associated to the topology, ie a grid
    where gridLnz[0] stores values of lnz and gridLnz[1] stores corresponding values of
    beta.
    Return :
     - estimated beta
     - tabulated distribution p(beta|labels)
    """
    if method == 'MAP':
        # p(beta | labels):
        pBeta = Cpt_Distrib_P_beta_graph(graph, labels, gridLnz[0], gridLnz[1])
        #print 'pBeta:', pBeta/pBeta.sum()
        #print 'gridLnz:', gridLnz[1]
        pBeta /= pBeta.sum()
        postMean = (gridLnz[1]*pBeta).sum()
        #varBetaEstim = (gridLnz[1]*gridLnz[1]*pBeta).sum() - postMean*postMean
        return postMean, pBeta
    elif method == 'ML':
        hc = count_homo_cliques(graph, labels, weights)
        #print 'hc:', hc
        logll = gridLnz[1] * hc - gridLnz[0]
        #print 'log likelihood:', logll.dtype
        #print logll
        pBeta = np.exp(logll.astype(float_hires))
        #print 'pBeta unnorm:', pBeta
        pBeta = pBeta/pBeta.sum()
        #print 'pBeta:', pBeta
        betaML = gridLnz[1][np.argmax(logll)]
        #dlpf = np.diff(lpf) / dbeta
        #gamma = dlpf/lpf[1:]
        #print 'gamma:', gamma
        #dbetaGrid = betaGrid[1:]
        #print 'dbetaGrid:', dbetaGrid
        #betaML = dbetaGrid[closestsorted(gamma, hc)]
        return betaML, pBeta
    else:
        raise Exception('Unknown method %s' %method)

#################
# Beta Sampling #
#################

class BetaSampler(xmlio.XMLParamDrivenClass, GibbsSamplerVariable):

    P_VAL_INI = 'initialValue'
    P_SAMPLE_FLAG = 'sampleFlag'
    P_USE_TRUE_VALUE = 'useTrueValue'

    P_PR_BETA_CUT = 'priorBetaCut'
    P_SIGMA = 'MH_sigma'

    P_PARTITION_FUNCTION = 'partitionFunction'
    P_PARTITION_FUNCTION_METH = 'partitionFunctionMethod'

    # parameters definitions and default values :
    defaultParameters = {
        P_SAMPLE_FLAG : True,
        P_USE_TRUE_VALUE : False,
        P_VAL_INI : np.array([0.7]),
        P_SIGMA : 0.05,
        P_PR_BETA_CUT : 1.2,
        P_PARTITION_FUNCTION_METH : 'es',
        P_PARTITION_FUNCTION : None,
        }

    if pyhrf.__usemode__ == pyhrf.DEVEL:
        parametersToSghow = [P_SAMPLE_FLAG, P_VAL_INI, P_SIGMA, P_PR_BETA_CUT,
                             P_USE_TRUE_VALUE,
                             P_PARTITION_FUNCTION, P_PARTITION_FUNCTION_METH,]
    elif pyhrf.__usemode__ == pyhrf.ENDUSER:
        parametersToShow = [P_SAMPLE_FLAG, P_VAL_INI]

    parametersComments = {
        P_PARTITION_FUNCTION_METH :  \
            'either "es" (extrapolation scheme) or "ps" (path sampling)',
        }
    #P_BETA : 'Amount of spatial correlation.\n Recommanded between 0.0 and'\
    #    ' 0.7',

    def __init__(self, parameters=None, xmlHandler=NumpyXMLHandler(),
                 xmlLabel=None, xmlComment=None):

        xmlio.XMLParamDrivenClass.__init__(self, parameters, xmlHandler,
                                           xmlLabel, xmlComment)
        sampleFlag = self.parameters[self.P_SAMPLE_FLAG]
        valIni = self.parameters[self.P_VAL_INI]
        useTrueValue = self.parameters[self.P_USE_TRUE_VALUE]
        an = ['condition']
        GibbsSamplerVariable.__init__(self,'beta', valIni=valIni,
                                      sampleFlag=sampleFlag,
                                      useTrueValue=useTrueValue,
                                      axes_names=an,
                                      value_label='PM Beta')
        self.priorBetaCut = self.parameters[self.P_PR_BETA_CUT]
        self.gridLnZ = self.parameters[self.P_PARTITION_FUNCTION]
        self.pfMethod = self.parameters[self.P_PARTITION_FUNCTION_METH]

        self.currentDB = None
        self.currentAcceptRatio = None


    def linkToData(self, dataInput):
        self.dataInput = dataInput
        nbc = self.nbConditions = self.dataInput.nbConditions
        self.sigma = np.zeros(nbc, dtype=float) + self.parameters[self.P_SIGMA]
        self.nbClasses = self.samplerEngine.getVariable('nrl').nbClasses
        self.nbVox = dataInput.nbVoxels

        self.pBeta = [ [] for c in xrange(self.nbConditions) ]
        self.betaWalk = [ [] for c in xrange(self.nbConditions) ]
        self.acceptBeta = [ [] for c in xrange(self.nbConditions) ]
        self.valIni = self.parameters[self.P_VAL_INI]

    def checkAndSetInitValue(self, variables):

        if self.useTrueValue :
            if self.trueValue is not None:
                self.currentValue = self.trueValue
            elif self.valIni is not None:
                self.currentValue = self.valIni
            else:
                raise Exception('Needed a true value for drift init but '\
                                    'None defined')
        if self.currentValue is not None:
            self.currentValue = np.zeros(self.nbConditions, dtype=float) \
                + self.currentValue[0]


    def samplingWarmUp(self, variables):
        if self.sampleFlag:
            self.loadBetaGrid()

    def loadBetaGrid(self):

        if self.gridLnZ is None:
            g = self.dataInput.neighboursIndexes
	    g = np.array([l[l!=-1] for l in g],dtype=object)
            if self.pfMethod == 'es':
                self.gridLnZ = Cpt_Vec_Estim_lnZ_Graph_fast3(g, self.nbClasses)
                pyhrf.verbose(2, 'lnz ES  ...')
                #print self.gridLnZ
            elif self.pfMethod == 'ps':
                self.gridLnZ = Cpt_Vec_Estim_lnZ_Graph(g, self.nbClasses)
                pyhrf.verbose(2, 'lnz PS  ...')
	    #HACK
	    #import matplotlib.pyplot as plt
	    #print 'nbClasses:', self.nbClasses
	    #print 'g:', g
	    #lnz_ps = Cpt_Vec_Estim_lnZ_Graph(g, self.nbClasses)
	    #lnz_es = Cpt_Vec_Estim_lnZ_Graph_fast3(g, self.nbClasses)

	    #plt.plot(lnz_es[1][:29],lnz_es[0][:29],'r-')
	    #plt.plot(lnz_ps[1],lnz_ps[0],'b-')

	    #plt.show()
	    #sys.exit(0)
	    #HACK


    def sampleNextInternal(self, variables):
        snrls = self.samplerEngine.getVariable('nrl')

        for cond in xrange(self.nbConditions):
            vlnz, vb = self.gridLnZ
            g = self.dataInput.neighboursIndexes
            labs = self.samplerEngine.getVariable('nrl').labels[cond,:]
            t = self.priorBetaCut
            b, db, a = Cpt_AcceptNewBeta_Graph(g, labs, vlnz, vb,
                                               self.currentValue[cond],
                                               self.sigma[0], thresh=t)
            self.currentDB = db
            self.currentAcceptRatio = a
            self.currentValue[cond] = b
            msg = "beta cond %d: %f" %(cond, self.currentValue[cond])
            pyhrf.verbose.printNdarray(5, msg)


    def get_string_value(self, v):
        return get_2Dtable_string(v, self.dataInput.cNames, ['pm_beta'])

    def saveCurrentValue(self, it):
        GibbsSamplerVariable.saveCurrentValue(self,it)
        if 0 and self.sampleFlag and self.currentDB is not None:
            for cond in xrange(self.nbConditions):
                vlnz, vb = self.gridLnZ
                g = self.dataInput.neighboursIndexes
                labs = self.samplerEngine.getVariable('nrl').labels[cond,:]
                t = self.priorBetaCut

                self.betaWalk[cond].append(self.currentDB)
                self.pBeta[cond].append(Cpt_Distrib_P_beta_graph(g, labs, vlnz,
                                                                 vb, thresh=t))
                self.acceptBeta[cond].append(self.currentAcceptRatio)

    def getOutputs(self):
        outputs = {}
        if pyhrf.__usemode__ == pyhrf.DEVEL:
            outputs = GibbsSamplerVariable.getOutputs(self)
            cn = self.dataInput.cNames
            axes_names = ['condition', 'voxel']
            axes_domains = {'condition' : cn}
            nbv, nbc = self.nbVox, self.nbConditions
            repeatedBeta = np.repeat(self.mean, nbv).reshape(nbc, nbv)
            outputs['pm_beta_mapped'] = xndarray(repeatedBeta,
                                               axes_names=axes_names,
                                               axes_domains=axes_domains,
                                               value_label="pm Beta")

            if self.sampleFlag:
                if self.pBeta is not None and len(self.pBeta[0])>0:
                    axes_names = ['condition', 'iteration', 'beta']
                    axes_domains = {'beta':self.gridLnZ[1], 'condition':cn,
                                   'iteration':self.smplHistoryIts[1:]}
                    pBeta = np.array(self.pBeta)
                    #print 'pBeta.shape:', pBeta.shape
                    outputs['pBetaApost'] = xndarray(pBeta, axes_names=axes_names,
                                                   value_label="proba",
                                                   axes_domains=axes_domains)
                if self.betaWalk is not None and len(self.betaWalk[0])>0:
                    axes_names = ['condition', 'iteration']
                    axes_domains = {'condition' : cn,
                                   'iteration':self.smplHistoryIts[1:]}
                    betaWalks = np.array(self.betaWalk)
                    #print 'betaWalk hist :', betaWalks.shape
                    outputs['betaWalkHist'] = xndarray(betaWalks,
                                                     axes_names=axes_names,
                                                     value_label="dBeta",
                                                     axes_domains=axes_domains)
                if self.acceptBeta is not None and len(self.acceptBeta[0])>0:
                    axes_names = ['condition', 'iteration']
                    axes_domains = {'condition' : cn,
                                   'iteration':self.smplHistoryIts[1:]}
                    acceptBetaHist = np.array(self.acceptBeta)
                    #print 'acceptBeta hist :', acceptBetaHist.shape
                    outputs['acceptBetaHist'] = xndarray(acceptBetaHist,
                                                       axes_names=axes_names,
                                                       value_label="Accept",
                                                       axes_domains=axes_domains)

                axes_names = ['beta']
                axes_domains = {'beta':self.gridLnZ[1]}
                nc = self.dataInput.nbCliques
                outputs['lnZ_div_NbCliques'] = xndarray(self.gridLnZ[0]/nc,
                                                      axes_names=axes_names,
                                                      value_label="lnZ/#cliques",
                                                      axes_domains=axes_domains)

        return outputs
