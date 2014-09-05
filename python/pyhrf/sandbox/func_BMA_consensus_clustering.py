# -*- coding: utf-8 -*-

import os
import numpy as np



def BMA_consensus_cluster_parallel(cfg, remote_path, remote_BOLD_fn, remote_mask_fn, Y, nifti_masker, \
                        num_vox, K_clus, K_clusters, \
                        parc, alpha, prop, nbItRFIR, onsets, durations,\
                        output_sub_parc, rescale=True, averg_bold=False):
    '''
    Performs all steps for one clustering case (Kclus given, number l of the parcellation given)
    remote_path: path on the cluster, where results will be stored
    '''
    import os
    import sys
    sys.path.append("/home/pc174679/pyhrf/pyhrf-tree_trunk/script/WIP/Scripts_IRMf_BB/Parcellations/")
    sys.path.append("/home/pc174679/pyhrf/pyhrf-tree_trunk/script/WIP/Scripts_IRMf_Adultes_Solv/")
    sys.path.append("/home/pc174679/pyhrf/pyhrf-tree_trunk/script/WIP/Scripts_IRMf_Adultes_Solv/Scripts_divers_utiles/Scripts_utiles/")
    sys.path.append('/home/pc174679/local/installations/consensus-cluster-0.6')
    
    from Random_parcellations import random_parcellations, subsample_data_on_time
    from Divers_parcellations_test import *
    
    from RFIR_evaluation_parcellations import JDE_estim, RFIR_estim, clustering_from_RFIR
    
    from Random_parcellations import hrf_roi_to_vox
    from pyhrf.tools._io import remote_copy, remote_mkdir
    from nisl import io
    
    #nifti_masker.mask=remote_mask_fn
    
    # Creation of the necessary paths --> do not do here
    parc_name = 'Subsampled_data_with_' + str(K_clus) + 'clusters' 
    parc_name_clus = parc_name + 'rnd_number_' + str(parc+1)
    
    remote_sub = os.sep.join((remote_path, parc_name))   
    #if not os.path.exists(remote_sub):
        #os.path.exists(remote_sub)
        #print 'remote_sub:', remote_sub
        #os.makedirs(remote_sub)
    remote_sub_parc = os.sep.join((remote_sub,parc_name_clus))   
    #if not os.path.exists(remote_sub_parc):
        #os.makedirs(remote_sub_parc)
    
    output_RFIR_parc = os.sep.join((output_sub_parc,'RFIR_estim'))
    
    ###################################
    ## 1st STEP: SUBSAMPLING
    print '--- Subsample data ---'
    Ysub = subsample_data_on_time(Y, remote_mask_fn, K_clus, alpha, prop, \
                    nifti_masker, rescale=rescale)
    print 'Ysub:', Ysub
    print 'remote_sub_prc:', remote_sub_parc
    Ysub_name = 'Y_sub_'+ str(K_clus) + 'clusters_' + 'rnd_number_' + str(parc+1) +'.nii'
    Ysub_fn = os.sep.join((remote_sub_parc, Ysub_name))
    Ysub_masked = nifti_masker.inverse_transform(Ysub).get_data()
    write_volume(Ysub_masked, Ysub_fn)                        
    
    
    
    ###################################
    ## 2D STEP: RFIR
    print '--- Performs RFIR estimation ---'

    
    remote_RFIR_parc_clus = os.sep.join((remote_sub_parc, 'RFIR_estim'))
    #if not os.path.exists(remote_RFIR_parc):os.makedirs(remote_RFIR_parc)
    #remote_RFIR_parc_clus = os.sep.join((remote_RFIR_parc, parc_name_clus))
    #if not os.path.exists(remote_RFIR_parc_clus):os.makedirs(remote_RFIR_parc_clus)
    
    print '  * output path for RFIR ', remote_RFIR_parc_clus
    print '  * RFIR for subsampling nb ', str(parc+1), ' with ', K_clus, ' clusters' 
    RFIR_estim(nbItRFIR, onsets, durations, Ysub_fn, remote_mask_fn, \
                remote_RFIR_parc, avg_bold=averg_bold) 
                  
    hrf_fn = os.sep.join((remote_RFIR_parc_clus, 'rfir_ehrf.nii'))
    #remote_copy([hrf_fn], remote_host, 
                #remote_user, remote_path)[0]
    
    ###################################
    ## 3D STEP: CLUSTERING FROM RFIR RESULTS
    name_hrf = 'rfir_ehrf.nii'
    
    from pyhrf.tools._io import write_volume, read_volume
    from pyhrf.tools._io import read_volume, write_volume
    import nisl.io as ionisl
    from sklearn.feature_extraction import image
    from sklearn.cluster import WardAgglomeration
    from scipy.spatial.distance import cdist, pdist
    
    hrf_fn = os.sep.join((remote_RFIR_parc_clus,name_hrf))
    hrf=read_volume(hrf_fn)[0]
    hrf_t_fn = add_suffix(hrf_fn, 'transpose')
    #taking only 1st condition to parcellate
    write_volume(hrf[:,:,:,:,0], hrf_t_fn)
    
    nifti_masker = ionisl.NiftiMasker(remote_mask_fn)
    Nm = nifti_masker.fit(hrf_t_fn)
    
    #features: coeff of the HRF
    HRF = Nm.fit_transform(hrf_t_fn)
    
    mask, meta_data = read_volume(remote_mask_fn)
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
            n_z=shape[2], mask=mask)
            
    #features used for clustering
    features = HRF.transpose()

    ward = WardAgglomeration(n_clusters=K_clus, connectivity=connectivity,
                                memory='nisl_cache')
    ward.fit(HRF)
    labels_tot = ward.labels_+1 
        
        
    #Kelbow, Perc_WSS, all_parc_from_RFIR_fns, all_parc_RFIR = \
    #clustering_from_RFIR(K_clusters, remote_RFIR_parc_clus, remote_mask_fn, name_hrf, plots=False)
    #labels_tot = all_parc_RFIR[str(Kelbow)]
    
    #to retrieve clustering with as many clusters as determined in K_clusters
    #labels_tot = all_parc_RFIR[str(K_clus)]
    #Parcellation retrieved: for K=Kelbow
    #clusters_RFIR_fn = all_parc_from_RFIR[str(Kelbow)]
    #clustering_rfir_fn = os.path.join(remote_RFIR_parc_clus, 'output_clustering_elbow.nii')
    #write_volume(read_volume(clusters_RFIR_fn)[0], clustering_rfir_fn, meta_bold)

    #labels_tot = nifti_masker.fit_transform([clusters_RFIR_fn])[0]
    #labels_tot = read_volume(clusters_RFIR_fn)[0]
    
    #labels_name='labels_' + str(int(K_clus)) + '_' + str(parc+1) + '.pck'
    #name_f = os.sep.join((remote_sub_parc, labels_name))
    #pickle_labels=open(name_f, 'w')
    #cPickle.dump(labels_tot,f)
    #pickle_labels.close()
    
    #remote_copy(pickle_labels, remote_user, 
            #remote_host, output_sub_parc)
    
    #################################
    ## Prepare consensus clustering
    print 'Prepare consensus clustering'
    clustcount, totalcount = upd_similarity_matrix(labels_tot)
    print 'results:', clustcount
    
    return clustcount.astype(np.bool)
    
    
    
    
    
    
def compute_consensus_clusters_parallel(K_clus, consensus_matrices, clustcount_matrices, \
        totalcount_matrices, num_voxels, remote_mask_fn, clusters_consensi):
    '''
    '''
    import nisl.io as ionisl
    import os
    import sys
    sys.path.append("/home/pc174679/pyhrf/pyhrf-tree_trunk/script/WIP/Scripts_IRMf_BB/Parcellations/")
    sys.path.append("/home/pc174679/pyhrf/pyhrf-tree_trunk/script/WIP/Scripts_IRMf_Adultes_Solv/")
    sys.path.append("/home/pc174679/pyhrf/pyhrf-tree_trunk/script/WIP/Scripts_IRMf_Adultes_Solv/Scripts_divers_utiles/Scripts_utiles/")
    sys.path.append('/home/pc174679/local/installations/consensus-cluster-0.6')
    
    from Random_parcellations import random_parcellations, subsample_data_on_time
    from Divers_parcellations_test import *
    from pyhrf.tools._io import read_volume
    
    print '  * Consensus with ', K_clus, ' clusters' 
    c_mat = np.array(clustcount_matrices[int(K_clus)])
    t_mat = np.array(totalcount_matrices[int(K_clus)])
    consensus_mat = gen_consensus_matrix(num_voxels, 0, c_mat, t_mat)
    
    parc_name = 'Subsampled_data_with_' + str(K_clus) + 'clusters' 
    output_sub = os.sep.join((output_path, parc_name)) 
    
    nifti_masker = ionisl.NiftiMasker(remote_mask_fn)
    mask_shape=read_volume(remote_mask_fn)[0].shape
    Nm = nifti_masker.fit(remote_mask_fn)
    #Nm = nifti_masker.fit(np.ones((mask_shape)))
 
    #clusterize the consensus matrix
    if clusterize_cons_mat:
        labels_cons_mat = hcluster_consensus(consensus_mat, num_voxels, K_clus, linkage='average')
        labels_cons_mat_inv = Nm.inverse_transform(labels_cons_mat).get_data()
      
    clusters_consensi[int(K_clus)] = np.zeros((K_clus))
    
    #compute cluster consensus, for each clustering
    clusters_consensi[int(K_clus)] = compute_cc_mat(K_clus, consensus_mat, labels_cons_mat, num_voxels)
            
            
    return clusters_consensi.astype('int32'), consensus_mat.astype('int32'), labels_cons_mat.astype('int16'), labels_cons_mat_inv.astype('int16')
    
    
    
    
    
    
    
