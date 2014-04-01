# -*- coding: utf-8 -*-


import os
import os.path as op
import re
import tempfile
from time import strftime, localtime, time
import numpy as np
from numpy.random import rand, randint, permutation


import pyhrf
from pyhrf.tools.io import read_volume, write_volume, read_texture, write_texture
from pyhrf.tools import format_duration, peelVolume3D
from pyhrf.graph import parcels_to_graphs, kerMask3D_6n, \
    connected_components
from pyhrf.ndarray import expand_array_in_mask
from pyhrf.glm import glm_nipy
from nipy.labs.spatial_models.parcel_io import fixed_parcellation

try:
    from nipy.algorithms.clustering.clustering import voronoi
except ImportError:
    from nipy.algorithms.clustering.utils import voronoi


def round_nb_parcels(n):
    if n >= 100:
        return int(np.round(n/100)*100)
    elif n >= 10:
        return  int(np.round(n/10)*10)
    else:
        return  int(np.round(n))


def parcellation_report(d):
    s = ''
    d = d.astype(int)
    s += 'Parcellation description:\n'
    roiIds = np.unique(d)
    m = d.min()
    id_shift = int(min(m,0))
    counts = np.bincount(d.astype(int).flatten()-id_shift)
    if len(roiIds) == 1:
        s+=  '* One ROI, label=%d, size=%d' %(roiIds[0]+id_shift,
                                              counts[roiIds[0]+id_shift])
        return s

    #avg_size = np.round(np.mean([(d==i).sum() for i in roiIds if i!=m]))
    if counts[0] > 0:
        avg_size = counts[counts!=0][1:].mean()
    else:
        avg_size = counts[counts!=0].mean()
    s +=  '* Nb voxels inside mask: %d\n' %(d>d.min()).sum()
    #print m, d.max(), avg_size
    s += '* %d parcels, min=%d, max=%d, avg parcel size=%d\n'  \
        %(len(roiIds), m, d.max(), avg_size)
    s += '* Size of background: %s\n' %counts[0]
    counts_tmp = counts.astype(float)
    counts_tmp[counts == 0] = np.inf
    s += '* Id of smallest parcel: %d (size=%d)\n' \
        %(np.argmin(counts_tmp)+id_shift, int(counts_tmp.min()))
    s += '* Nb of parcels with size < 10: %d\n' %(counts_tmp<10).sum()
    #s += '* Id of biggest parcel: %d (size=%d)\n' \
    #    %(np.argmax(counts)+id_shift, counts.max())
    if 0:
        print m-id_shift+1
        print zip(range(len(counts)),counts)
        print counts[1:].max()
        print sorted(counts)
    argmax = np.argmax(counts[1:]) + 1
    s += '* Id of biggest parcel: %d (size=%d)\n' \
        %(argmax + id_shift , counts[argmax])
    assert (d==(argmax + id_shift)).sum() == counts[argmax]
    s += '* Nb of parcels with size is > 400: %d\n' \
        %(counts[m-id_shift+1:]>400).sum()
    s += '* Nb of parcels with size is > 1000: %d\n' \
        %(counts[m-id_shift+1:]>1000).sum()

    return s

def parcellation_ward_spatial(func_data, n_clusters, graph=None):
    """
    Make parcellation based upon ward hierarchical clustering from scikit-learn
    Inputs: - func_data: functional data: array of shape
              (nb_positions, dim_feature_1, [dim_feature2, ...])
            - n_clusters: chosen number of clusters to create
            - graph: adjacency list defining neighbours
              (if None, no connectivity defined: clustering is spatially
            independent)
    Output: parcellation labels
    """
    from sklearn.cluster import Ward
    from pyhrf.graph import graph_to_sparse_matrix, sub_graph
    #from pyhrf.tools import cartesian
    # print 'graph:'
    # print graph
    if graph is not None:

        #print 'connectivity;'
        #print connectivity.todense()
        labels = np.zeros(len(graph), dtype=np.int32)

        ccs = connected_components(graph)
        n_tot = len(graph) * 1.

        ncs = np.zeros(len(ccs), dtype=np.int32)
        for icc,cc in enumerate(ccs):
            nc = int(np.round(n_clusters * len(cc)/n_tot))
            if nc == 0:
                nc = 1
            ncs[icc] = nc

        max_nc = np.argmax(ncs)
        ncs[max_nc] = n_clusters - sum(ncs[0:max_nc]) - sum(ncs[max_nc+1:])

        assert sum(ncs) == n_clusters
        assert (ncs > 0).all()

        pyhrf.verbose(1, 'Found %d connected components (CC) of sizes: %s' \
            %(len(ccs), ' ,'.join([str(len(cc)) for cc in ccs])))
        pyhrf.verbose(1, 'Nb of clusters to search in each CC: %s' \
                        %' ,'.join(map(str,ncs)))

        for nc,cc in zip(ncs,ccs):
            #print 'cc:', len(cc)
            #print 'cartesian:', list(cartesian(cc,cc))
            if len(cc) < 2:
                continue

            if len(cc) < len(graph):
                cc_graph,_ = sub_graph(graph, cc)
            else:
                cc_graph = graph

            cc_connectivity = graph_to_sparse_matrix(cc_graph)

                #print 'compute subslice ...'
#                sub_slice = tuple(np.array(list(cartesian(cc,cc))).T) #indexes of the subpart of the connectivity matrix
                #print 'sub_slice:'
                #print sub_slice
                #print 'subslice connectivity matrix ...'
                #cc_connectivity = connectivity.tolil()[sub_slice].reshape((len(cc),len(cc))).tocoo() #indexes applied to the matrix connectivity (coo unsubscriptable)
            #print 'cc_connectivity'
            #print cc_connectivity.todense()
            cc_data = func_data[cc]
            pyhrf.verbose(2, 'Launch spatial Ward (nclusters=%d) '\
                              ' on data of shape %s' %(nc, str(cc_data.shape)))
            ward_object = Ward(n_clusters=nc, connectivity=cc_connectivity).fit(cc_data)
            labels[cc] += ward_object.labels_ + 1 + labels.max()
    else:
        ward_object = Ward(n_clusters=n_clusters).fit(func_data) # connectivity=None
        labels = ward_object.labels_ + 1

    return labels



def make_parcellation_from_files(betaFiles, maskFile, outFile, nparcels,
                                 method, dry=False, spatial_weight=10.):

    if not op.exists(maskFile):
        print 'Error, file %s not found' %maskFile
        return

    betaFiles = sorted(betaFiles)
    for b in betaFiles:
        if not op.exists(b):
            raise Exception('Error, file %s not found' %b)

    pyhrf.verbose(1, 'Mask image: ' + op.basename(maskFile))
    pyhrf.verbose(1, 'Betas: ' + op.basename(betaFiles[0]) + ' ... ' + \
                      op.basename(betaFiles[-1]))
    pyhrf.verbose(1, "Method: %s, nb parcels: %d" %(method, nparcels))
    pyhrf.verbose(1, 'Spatial weight: %f' %spatial_weight)

    if not dry:
        pyhrf.verbose(1, 'Running parcellation ... ')
        pyhrf.verbose(1, 'Start date is: %s' %strftime('%c',localtime()))
        t0 = time()
        #lpa = one_subj_parcellation(maskFile, betaFiles, nparcels, 6,
        #                            method, 10, 1, fullpath=outFile)
        v = pyhrf.verbose.verbosity
        lpa = fixed_parcellation(maskFile, betaFiles, nparcels, nn=6,
                                 method=method, fullpath=outFile, verbose=v,
                                 mu=spatial_weight)

        from pyhrf.ndarray import xndarray
        c = xndarray.load(outFile)
        if c.min() == -1:
            c.data += 1

        for i in np.unique(c.data):
            # remove parcel with size < 2:
            if i!=0 and len(c.data==1) < 2:
                c.data[np.where(c.data==i)] = 0

        c.save(outFile)

        pyhrf.verbose(1, 'Parcellation complete, took %s' \
                          %format_duration(time() - t0))
        return lpa
    else:
        pyhrf.verbose(1, 'Dry run.')


from nipy.labs.spatial_models.discrete_domain import domain_from_mesh
from nipy.algorithms.graph.field import field_from_coo_matrix_and_data
#from nipy.algorithms.clustering.utils import kmeans
from nipy.labs.spatial_models.mroi import SubDomains

def make_parcellation_surf_from_files(beta_files, mesh_file, parcellation_file,
                                      nbparcel, method, mu=10., verbose=0):

    if method not in ['ward', 'gkm', 'ward_and_gkm', 'kmeans']:
        raise ValueError('unknown method')


    # step 1: load the data ----------------------------
    # 1.1 the domain
    pyhrf.verbose(3, 'domain from mesh: %s' %mesh_file)
    domain = domain_from_mesh(mesh_file)

    coord = domain.coord

    # 1.3 read the functional data
    beta = np.array([read_texture(b)[0] for b in beta_files]).T

    pyhrf.verbose(3, 'beta: %s' %str(beta.shape))
    pyhrf.verbose(3, 'mu * coord / np.std(coord): %s' \
                      %(mu * coord / np.std(coord)).shape)
    feature = np.hstack((beta, mu * coord / np.std(coord)))

    if method is not 'kmeans':
        # print 'domain.topology:', domain.topology.__class__
        # print domain.topology
        #print dir(domain.topology)
        # print 'feature:', feature.shape
        # print feature
        g = field_from_coo_matrix_and_data(domain.topology, feature)
        # print 'g:', g.__class__
        # print g


    if method == 'kmeans':
        _, u, _ = kmeans(feature, nbparcel)

    if method == 'ward':
        u, _ = g.ward(nbparcel)

    if method == 'gkm':
        seeds = np.argsort(np.random.rand(g.V))[:nbparcel]
        _, u, _ = g.geodesic_kmeans(seeds)

    if method == 'ward_and_gkm':
        w, _ = g.ward(nbparcel)
        _, u, _ = g.geodesic_kmeans(label=w)

    # print 'u:'
    # print u

    lpa = SubDomains(domain, u, 'parcellation')

    if verbose:
        var_beta = np.array(
            [np.var(beta[lpa.label == k], 0).sum() for k in range(lpa.k)])
        var_coord = np.array(
            [np.var(coord[lpa.label == k], 0).sum() for k in range(lpa.k)])
        size = lpa.get_size()
        vf = np.dot(var_beta, size) / size.sum()
        va = np.dot(var_coord, size) / size.sum()
        print nbparcel, "functional variance", vf, "anatomical variance", va

    # step3:  write the resulting label image
    if parcellation_file is not None:
        label_image = parcellation_file
    # elif write_dir is not None:
    #     label_image = os.path.join(write_dir, "parcel_%s.nii" % method)
    else:
        label_image = None

    if label_image is not None:
        #lpa.to_image(label_image, descrip='Intra-subject parcellation image')
        write_texture(u.astype(np.int32), label_image)
        if verbose:
            print "Wrote the parcellation images as %s" % label_image

    return u, label_image

    #return lpa



#############################################################
## Balanced partitioning using balloon-like a(ge)nt system ##
#############################################################


def random_pick(a):
    return a[randint(0,a.size)]

def init_edge_data(g, init_value=0):
    return [dict([(i,init_value) for i in nl]) for nl in g]

class Talker:

    def __init__(self, talker_string_id, verbosity=0):
        self.verbosity = verbosity
        if len(talker_string_id) > 0:
            self.t_id = '%s ||' %talker_string_id
        else:
            self.t_id = ''

    def verbose(self, level, msg):
        if self.verbosity >= level:
            stars = '*' * level
            print stars, self.t_id, msg

    def verbose_array(self, level, array):
        if self.verbosity >= level:
            stars = '*' * level
            if array.ndim > 1:
                for l in array:
                    print stars, self.t_id, l
            else:
                print stars, self.t_id, array


def Visit_graph_noeud(noeud, graphe, Visited=None):
    if Visited is None:
        Visited=np.zeros(len(graphe), dtype = int)

    Visited[noeud] = 1
    for voisin in graphe[noeud]:
        if not Visited[voisin]:
            Visit_graph_noeud(voisin, graphe, Visited)

    return Visited



class World(Talker):

    def __init__(self, graph, nb_ants, greed=.05,
                 time_min=100, time_max=None, tolerance=1, verbosity=0,
                 stop_when_all_controlled=False):


        #Talker.__init__(self, 'World', verbosity)
        Talker.__init__(self, '', verbosity)
        self.tolerance = tolerance
        self.time_max = time_max
        self.time_min = time_min
        self.nb_ants = nb_ants
        # self.valid_positions = valid_positions
        # if valid_positions.ndim == 2:
        #     self.km = kerMask2D_4n
        # elif valid_positions.ndim == 3:
        #     self.km = kerMask3D_6n
        # else:
        #     self.km = None

        #self.graph = graph_from_lattice(valid_positions, self.km)
        self.graph = graph

        #print 'self.graph:', type(self.graph), type(self.graph[0])
        #print self.graph.dtype
        #print self.graph
        self.nb_sites = len(self.graph)
        self.labels = np.zeros(self.nb_sites, dtype=np.int32) - 1
        self.path_marks = init_edge_data(self.graph)
        self.site_marks = np.zeros(self.nb_sites,dtype=np.int32)
        self.pressures = init_edge_data(self.graph)

        self.ants = [Ant(i, greed, self.graph, self.labels, self.path_marks,
                         self.site_marks, self.pressures, self, verbosity)
                     for i in xrange(self.nb_ants)]

        self.uncontrolled = self.nb_sites - self.nb_ants

        self.tmark = time()
        self.prev_uncontrolled = self.nb_sites
        self.total_time = .0

        self.stop_at_full_control = stop_when_all_controlled

    def balanced(self):

        ##self.verbose(1,'Checking balance ...')
        ##self.verbose(1,'Uncontrolled: %d/%d' \
        ##                 %(self.uncontrolled,self.nb_sites))
        #self.verbose(1,'True uncontrolled: %d/%d' \
        #                 %(sum(self.labels==-1),self.nb_sites))
        #assert sum(self.labels==-1) == self.uncontrolled
        if self.uncontrolled > 0:
            return False

        if self.time < self.time_min:
            self.verbose(6,'time_min not reached')
            return False

        # territory_sizes = [a.area_size for a in self.ants]
        # self.verbose(1,'Territory sizes: %s' %str(territory_sizes))
        # self.verbose(1,'Total controlled  territories: %d / %d'  \
        #                  %(sum(territory_sizes), self.nb_sites))
        # if sum(territory_sizes) != self.nb_sites:
        #     return False


        return np.allclose([a.area_size for a in self.ants],
                           self.nb_sites/self.nb_ants, atol=self.tolerance)
            #t0 = time()
            #e = self.balloon_explosion()
            #print 'explosion check took %s' %format_duration(time()-t0)
            #return not e

        # else:
        #     return False


    # def balloon_explosion(self):
    #     ##self.verbose(1,'Check for explosion ...')
    #     current_labels = np.zeros_like(self.valid_positions)
    #     current_labels[np.where(self.valid_positions)] = self.labels
    #     for ant_id in xrange(self.nb_ants):
    #         ##self.verbose(2,'Test subgraph of ant %d ...' %ant_id)
    #         g = graph_from_lattice(current_labels==ant_id, self.km)
    #         nb_comps = nb_connected_components(g)
    #         ##self.verbose(2,'Nb connected components: %d' %nb_comps)
    #         if nb_comps > 1:
    #             return True
    #     return False


    def force_end(self):
        if self.stop_at_full_control and self.uncontrolled == 0:
            return True

        if self.time_max is None:
            return False
        else:
            return self.time >= self.time_max

    def site_taken(self, site):
        #print 'Recieved notification of taken site %d' %site
        ant_mark = self.labels[site]
        #print '-> mark', ant_mark
        if ant_mark != -1:
            #print 'decrease size accordingly'
            self.ants[ant_mark].territory_size -= 1

    def resolve(self):

        t0 = time()
        self.time = 0
        while not self.balanced() and not self.force_end():
            ##self.verbose(1,'---------- iteration %d ----------' %self.time)
            for a_id in permutation(self.nb_ants):
                if self.ants[a_id].action(self.time):
                    continue
            self.time += 1
            if self.time % 100000 == 0:
                self.verbose(1,'Time: %d' %self.time)

                self.verbose(1,'Nb of uncontrolled sites: %d' %self.uncontrolled)
                sizes = [str(a.area_size) for a in self.ants]
                self.verbose(1,'Territories: %s' %','.join(sizes))
                if self.uncontrolled > 0:
                    delta_t = time() - self.tmark
                    duration = format_duration(delta_t)
                    delta_site = self.prev_uncontrolled - self.uncontrolled
                    self.verbose(1,'%d sites taken in %s' \
                                     %(delta_site, duration))
                    self.total_time += delta_t
                    speed = (self.nb_sites - self.uncontrolled) / self.total_time
                    expected_duration = self.uncontrolled / speed
                    self.verbose(1,'%s expected for complete control' \
                                     %(format_duration(expected_duration)))

                    self.prev_uncontrolled = self.uncontrolled
                    self.tmark = time()

            ##self.verbose_array(1,self.labels)
            ##self.verbose(1,'')
            ##self.verbose(1,'----------------------------------')
        self.verbose(1,'Balloon partitioning done, took %s, %d iterations'
                     %(format_duration(time()-t0), self.time))

    def get_final_labels(self):
        return self.labels
        ##self.verbose(6,'Final labels:')
        ##self.verbose_array(6,final_labels)
        ##self.verbose(0,'Territories: '+str([a.area_size for a in self.ants]))

class Ant(Talker):

    def __init__(self, a_id, greed, graph, labels, path_marks, site_marks,
                 pressures, world, verbosity=0):
        Talker.__init__(self, 'ant %d'%a_id, verbosity)
        self.world = world
        self.search_level = 0
        self.area_size = 1
        self.labels = labels
        self.path_marks = path_marks
        self.site_marks = site_marks
        self.pressures = pressures
        self.graph = graph
        self.a_id = a_id
        self.greed = greed
        self.verbosity = verbosity
        #self.territory_size = 1

        # pick init position
        if self.labels[0] == -1:
            start_pos = 0
        elif self.labels[-1] == -1:
            start_pos = len(self.labels)-1
        elif self.labels[len(self.labels)/2] == -1:
            start_pos = len(self.labels)/2
        else:
            unmarked = np.where(self.labels==-1)
            picked = randint(len(unmarked[0]))
            start_pos = unmarked[0][picked]

        self.labels[start_pos] = a_id

        #print 'labels after init marking:'
        #print self.labels
        self.u = np.where(self.labels == self.a_id)[0][0] # current position


    def to_conquer(self):
        #print '~~~~~~~~'
        #print self.graph[self.u]
        for v in self.graph[self.u]:
            #print v
            #print self.path_marks[self.u][v]
            if self.labels[v] != self.a_id and \
                    self.path_marks[self.u][v] < self.search_level:
                yield v

    def to_patrol(self):
        for v in self.graph[self.u]:
            if self.labels[v] == self.a_id and \
                    self.path_marks[self.u][v] < self.search_level:
                yield v

    def fix_explosion(self):
        if self.world.uncontrolled < 1:
            for v in self.graph[self.u]:
                if self.labels[v] != self.a_id and self.labels[v] != -1:
                    orphan = True
                    for w in self.graph[v]:
                        if self.labels[w] != -1 and \
                                self.labels[w] == self.labels[v]:
                            orphan = False
                            break
                    if orphan:
                        #print v,'is oprhan'
                        self.labels[v] = -1
                        self.world.uncontrolled += 1
                        return

    def action(self,time):

        if self.labels[self.u] != self.a_id:
            ##self.verbose(1,"Current site  (%d) not owned -> %d!" \
            ##                 %(self.u,self.labels[self.u]))
            owned_neighbours = self.labels[self.graph[self.u]] == self.a_id
            if owned_neighbours.sum() > 0:
                self.u = random_pick(self.graph[self.u][owned_neighbours])
                ##self.verbose(1,"Moving to -> %d!" %(self.u))
                return False

        # Try to conquer not owned nearby sites:
        ##self.verbose(1,"Conquest stage")
        for v in self.to_conquer():
            # activate exploration of path:
            self.path_marks[self.u][v] = time
            ##self.verbose(2,"Seeing path %d->%d" %(self.u,v))
            if self.pressures[self.u][v] > 0:
                if rand() < self.greed:
                    ##self.verbose(2,"Attack of site %d succeeded" %v)
                    for w in self.graph[v]:
                        self.path_marks[v][w] = time + 1
                        self.pressures[v][w] = 0
                        self.pressures[w][v] = 0
                    self.path_marks[v][self.u] = time
                    self.site_marks[v] = time
                    ##self.verbose(2,'area_size: %d' %self.area_size)
                    #self.world.site_taken(v)
                    #self.territory_size += 1
                    if self.labels[v] == -1:
                        self.world.uncontrolled -= 1
                    self.labels[v] = self.a_id
                    self.u = v
                    self.fix_explosion()
                    return False
                else:
                    pass
                    ##self.verbose(2,"Attack of site %d failed" %v)
            else:
                if self.pressures[self.u][v] + self.area_size == -1:
                    self.pressures[self.u][v] = 0
                    self.pressures[v][self.u] = 0
                else:
                    self.pressures[self.u][v] = self.area_size
                    self.pressures[v][self.u] = -self.area_size

        # Partrol in nearby owned site
        ##self.verbose(1,"Patrol stage")
        for v in self.to_patrol():
            self.path_marks[self.u][v] = time
            if self.site_marks[v] < self.search_level:
                self.area_size = (time - self.site_marks[v] + 1) / 2
                self.path_marks[v][self.u] = time
                self.site_marks[v] = time
                ##self.verbose(2,"Patrolling to site %d" %v)
                self.u = v
                return False

        for v in self.graph[self.u]:
            if self.labels[v] == self.a_id and \
                    self.site_marks[self.u] > self.search_level and \
                    self.path_marks[self.u][v] == self.site_marks[self.u]:
                ##self.verbose(1,"Moving to site %d" %v)
                self.u = v
                return False

        ##self.verbose(1,"idle")
        # Agent didn't do anything
        self.search_level = time
        self.site_marks[self.u] = time
        for w in self.graph[self.u]:
            self.path_marks[self.u][w] -= 1

        return True


import pyhrf.graph as mg

def parcellate_balanced_vol(mask, nb_parcels):
    """
    Performs a balanced partitioning on the input mask using a balloon patroling
     algorithm [Eurol 2009]. Values with 0 are discarded position in the mask.

    Args:
        - mask (numpy.ndarray): binary 3D array of valid position to parcellate
        - nb_parcels (int): the required number of parcels

    Return:
        - the parcellation (numpy.ndarray): a 3D array of integers
    """

    parcellation = np.zeros(mask.shape, dtype=int)
    nvox = (mask != 0).sum()

    # Iterate over connected components in the input mask
    for cc_mask in mg.split_mask_into_cc_iter(mask != 0):
        pyhrf.verbose(2, 'Treating a connected component (CC) of %d positions' \
                          %cc_mask.sum())
        g = mg.graph_from_lattice(cc_mask)
        size_cc = cc_mask.sum()
        # compute the required number of parcels within the current CC:
        cc_np = max(int(np.round(nb_parcels * size_cc / (nvox*1.))), 1)

        pyhrf.verbose(2, 'Split (CC) into %d parcels' %cc_np)

        cc_labels = np.ones(cc_mask.sum(), dtype=int)
        if cc_np > 1:
            split_parcel(cc_labels, {1:g}, 1, cc_np, inplace=True,
                         verbosity=2, balance_tolerance='draft')
        else: #only one parcel expected, CC must be too small to be splited
            cc_labels[:] = 1
        pyhrf.verbose(2, 'Split done!')

        # accumulate parcellation result
        maxp = parcellation.max()
        parcellation += expand_array_in_mask(cc_labels + maxp, cc_mask>0)

    return parcellation

def split_parcel(labels, graphs, id_parcel, n_parcels, inplace=False,
                 verbosity=0, balance_tolerance='exact'):
    """
    balance_tolerance : exact or draft
    """
    #print 'split_parcel...'
    g = graphs[id_parcel]
    nb_cc = len(connected_components(g))
    if nb_cc > 1:
        raise Exception("The input graph is not connex " \
                            "(%d connected components found)" %nb_cc)
    #print 'g', len(g)
    #print g
    #print 'labels', len(labels)
    #print labels
    world = World(g, n_parcels, verbosity=verbosity,
                  stop_when_all_controlled=(balance_tolerance=='draft'))
    world.resolve()

    fl = world.get_final_labels()
    fl[fl>0] += labels.max()
    fl[fl==0] = id_parcel

    if not inplace:
        new_labels = labels.copy()
        input_labels = labels
    else:
        input_labels = labels.copy()
        new_labels = labels

    if 0:
        print 'new_labels:', new_labels.shape
        print 'input_labels:', input_labels.shape, new_labels.dtype
        print 'id_parcel:', id_parcel
        print 'new_labels[np.where(input_labels==id_parcel)]:', \
            new_labels[np.where(input_labels==id_parcel)].shape
        print np.unique(new_labels[np.where(input_labels==id_parcel)])
        print 'fl:', fl.shape, fl.dtype
        print np.unique(fl)

    new_labels[np.where(input_labels==id_parcel)] = fl
    return new_labels


def split_big_parcels(parcel_file, output_file, max_size=400):
    print 'split_big_parcels ...'
    roiMask, roiHeader = read_volume(parcel_file)
    roiIds = np.unique(roiMask)
    background = roiIds.min()
    labels = roiMask[np.where(roiMask>background)].astype(int)
    if (np.bincount(labels) <= max_size).all():
        pyhrf.verbose(1, 'no parcel to split')
        return

    graphs = parcels_to_graphs(roiMask, kerMask3D_6n)
    for roiId in roiIds:
        if roiId != background:
            roi_size = (roiMask==roiId).sum()
            if roi_size > max_size:
                print 'roi %d, size = %d' %(roiId, roi_size)
                nparcels = int(np.ceil(roi_size*1./max_size))
                print 'split into %d parcels ...' %(nparcels)
                split_parcel(labels, graphs, roiId, nparcels, inplace=True,
                             verbosity=1)

    final_roi_mask = np.zeros_like(roiMask)
    final_roi_mask[np.where(roiMask>background)] = labels
    #print np.bincount(labels)
    assert (np.bincount(labels) <= max_size).all()
    write_volume(final_roi_mask, output_file, roiHeader)



def parcellate_voronoi_vol(mask, nb_parcels, seeds=None):
    """
    Produce a parcellation from a Voronoi diagram built on random seeds.
    The number of seeds is equal to the nb of parcels.
    Seed are randomly placed within the mask, expect on edge positions

    Args:
        - mask (numpy.ndarray): binary 3D array of valid position to parcellate
        - nb_parcels (int): the required number of parcels
        - seeds: TODO

    Return:
        - the parcellation (numpy.ndarray): a 3D array of integers
        -
    """
    parcellation = np.zeros(mask.shape, dtype=int)
    nvox = (mask !=0).sum()
    for cc_mask in mg.split_mask_into_cc_iter(mask!=0):
        # compute the required number of parcels within the current CC:
        size_cc = cc_mask.sum()
        cc_np = max(int(np.round(nb_parcels * size_cc / (nvox*1.))), 1)
        pyhrf.verbose(2, 'Treating a connected component (CC) of %d positions' \
                          %cc_mask.sum())
        if cc_mask.sum() < 6:
            continue
        if seeds is None:
            # perform voronoi on random seeds

            eroded_mask = peelVolume3D(cc_mask)
            eroded_mask_size = eroded_mask.sum()
            if eroded_mask_size < nb_parcels: #do no erode, mask too small
                eroded_mask_size = nvox
                eroded_mask = mask.copy()
            cc_seeds = np.random.randint(0,eroded_mask_size, cc_np)
            mask_for_seed = np.zeros(eroded_mask_size, dtype=int)
            mask_for_seed[cc_seeds] = 1
            mask_for_seed  = expand_array_in_mask(mask_for_seed, eroded_mask)
        else:
            mask_for_seed = seeds * cc_mask

        pyhrf.verbose(2, 'Nb of seeds in current CC: %d' \
                          %mask_for_seed.sum())
        cc_parcellation = voronoi(np.vstack(np.where(cc_mask)).T,
                                  np.vstack(np.where(mask_for_seed)).T) + 1
        pyhrf.verbose(3, 'CC parcellation labels: %s' \
                          %str(np.unique(cc_parcellation)))
        maxp = parcellation.max()
        parcellation += expand_array_in_mask(cc_parcellation + maxp, cc_mask)
        pyhrf.verbose(3, 'Current parcellation labels: %s' \
                          %str(np.unique(parcellation)))
    pyhrf.verbose(1, 'voronoi parcellation: %s, %s' \
                      %(str(parcellation.shape), str(parcellation.dtype)))

    return parcellation


def parcellation_for_jde(fmri_data, avg_parcel_size=250, output_dir=None,
                         method='gkm', glm_drift='Cosine', glm_hfcut=128):
    """
    method: gkm, ward, ward_and_gkm
    """

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='pyhrf_JDE_parcellation_GLM',
                                      dir=pyhrf.cfg['global']['tmp_path'])
    glm_output_dir = op.join(output_dir, 'GLM_for_parcellation')
    if not op.exists(glm_output_dir): os.makedirs(glm_output_dir)

    pyhrf.verbose(1, 'GLM for parcellation')

    # if fmri_data.data_type == 'volume':
    #     paradigm_file, bold_file, mask_file = fmri_data.save(glm_output_dir)
    #     beta_files = glm_nipy_from_files(bold_file, fmri_data.tr, paradigm_file,
    #                                      glm_output_dir, mask_file,
    #                                      drift_model=glm_drift, hfcut=glm_hfcut)
    # elif fmri_data.data_type == 'surface':
    #     beta_files = glm_nipy(fmri_data, glm_output_dir,
    #                           drift_model=glm_drift, hfcut=glm_hfcut)

    g, dm, cons = glm_nipy(fmri_data, drift_model=glm_drift, hfcut=glm_hfcut)

    pval_files = []
    if cons is not None:
        func_data = [('con_pval_%s' %cname, con.pvalue()) \
                         for cname, con in cons.iteritems()]
    else:
        reg_cst_drift = re.compile(".*constant.*|.*drift.*")
        func_data = [('beta_%s' %reg_name, g.beta[ir]) \
                         for ir,reg_name in enumerate(dm.names) \
                         if not reg_cst_drift.match(reg_name)]

    for name, data in func_data:
        val_vol = expand_array_in_mask(data, fmri_data.roiMask>0)
        val_fn = op.join(glm_output_dir, '%s.nii' %name)
        write_volume(val_vol, val_fn, fmri_data.meta_obj)
        pval_files.append(val_fn)

    mask_file = op.join(glm_output_dir,'mask.nii')
    write_volume(fmri_data.roiMask>0, mask_file, fmri_data.meta_obj)

    nvox = fmri_data.get_nb_vox_in_mask()
    nparcels = round_nb_parcels(nvox * 1. / avg_parcel_size)

    pyhrf.verbose(1, 'Parcellation from GLM outputs, method: %s, ' \
                      'nb parcels: %d' %(method, nparcels))

    if fmri_data.data_type == 'volume':
        parcellation_file = op.join(output_dir, 'parcellation_%s_np%d.nii'
                                    %(method, nparcels))

        make_parcellation_from_files(pval_files, mask_file, parcellation_file,
                                     nparcels, method)
        parcellation,_ = read_volume(parcellation_file)
    else:
        mesh_file = fmri_data.data_files[-1]
        parcellation_file = op.join(output_dir, 'parcellation_%s_np%d.gii'
                                    %(method, nparcels))
        make_parcellation_surf_from_files(pval_files, mesh_file,
                                          parcellation_file, nparcels, method,
                                          verbose=1)
        parcellation,_ = read_texture(parcellation_file)
    #print parcellation_file


    pyhrf.verbose(1, parcellation_report(parcellation))

    return parcellation, parcellation_file


def make_parcellation_cubed_blobs_from_file(parcellation_file, output_path,
                                            roi_ids=None, bg_parcel=0,
                                            skip_existing=False):


    p,mp = read_volume(parcellation_file)
    p = p.astype(np.int32)
    if bg_parcel==0 and p.min() == -1:
        p += 1 #set background to 0

    if roi_ids is None:
        roi_ids = np.unique(p)

    pyhrf.verbose(1,'%d rois to extract' %(len(roi_ids)-1))

    tmp_dir = pyhrf.get_tmp_path('blob_parcellation')
    tmp_parcel_mask_file = op.join(tmp_dir, 'parcel_for_blob.nii')

    out_files = []
    for roi_id in roi_ids:
        if roi_id != bg_parcel: #discard background
            output_blob_file = op.join(output_path, 'parcel_%d_cubed_blob.arg'\
                                           %roi_id)
            out_files.append(output_blob_file)
            if skip_existing and os.path.exists(output_blob_file):
                continue
            parcel_mask = (p==roi_id).astype(np.int32)
            write_volume(parcel_mask, tmp_parcel_mask_file, mp)
            pyhrf.verbose(3,'Extract ROI %d -> %s' %(roi_id,output_blob_file))
            cmd = 'AimsGraphConvert -i %s -o %s --bucket' \
                %(tmp_parcel_mask_file, output_blob_file)
            pyhrf.verbose(3,'Cmd: %s' %(cmd))
            os.system(cmd)
    if op.exists(tmp_parcel_mask_file):
        os.remove(tmp_parcel_mask_file)

    return out_files


from pyhrf.cparcellation import compute_intersection_matrix


def parcellation_dist(p1, p2, mask=None):
    """
    Compute the distance between the two parcellation p1 and p2 as the minimum
    number of positions to remove in order to obtain equal partitions.
    "mask" may be a binary mask to limit the distance computation to some
    specific positions.
    Important convention: parcel label 0 is treated as background and
    corresponding positions are discarded if no mask is provided.

    Return:
        (distance value, parcellation overlap)
    """
    assert np.issubdtype(p1.dtype, np.int)
    assert np.issubdtype(p2.dtype, np.int)

    assert p1.shape == p2.shape

    from munkres import Munkres
    if mask is None:
        mask = (p1 != 0)


    m = np.where(mask)
    pyhrf.verbose(6,'Nb pos inside mask: %d' %len(m[0]))

    fp1 = p1[m].astype(np.int32)
    fp2 = p2[m].astype(np.int32)

    # allocate cost matrix, assume that region labels are contiguous
    # ie all labels in [1, label_max] are represented
    cost_matrix = np.zeros((fp1.max()+1, fp2.max()+1), dtype=np.int32)

    pyhrf.verbose(6,'Cost matrix : %s' %str(cost_matrix.shape))
    compute_intersection_matrix(fp1, fp2, cost_matrix)

    # discard 0-labelled parcels (background)
    cost_matrix = cost_matrix[1:,1:]

    # solve the assignement problem:
    indexes = np.array(Munkres().compute((cost_matrix*-1).tolist()))

    if 0:
        print 'assignement indexes:'
        print indexes
        print '->'
        print (indexes[:,0], indexes[:,1])
        print cost_matrix[(indexes[:,0], indexes[:,1])]

    assignement = cost_matrix[(indexes[:,0], indexes[:,1])].sum()

    to_keep = np.zeros_like(fp1)
    for s1, s2 in indexes:
        to_keep += np.bitwise_and(fp1==(s1+1), fp2==(s2+1))

    return fp1.size - assignement, expand_array_in_mask(to_keep, mask)




