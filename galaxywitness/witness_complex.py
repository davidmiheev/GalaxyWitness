import math
import os

import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

# gudhi is needed to construct a simplex tree and to plot the persistence diagram.
try:
    import gudhi
    from gudhi.clustering.tomato import Tomato
except:
    print('Failed to import gudhi')
from sklearn.metrics import pairwise_distances

# hard-coded
#MAX_DIST_INIT = 100000
MAX_N_PLOT = 5000
NUMBER_OF_FRAMES = 6

class WitnessComplex():
    __slots__ = [
        'landmarks',
        'witnesses',
        'distances',
        'distances_isomap',
        'landmarks_idxs',
        'isomap_eps',
        'landmarks_dist',
        'simplex_tree',
        'simplex_tree_computed',
        'metric_computed'
    ]

    def __init__(self, landmarks, witnesses, landmarks_idxs, n_jobs = 1, isomap_eps = 0):
        #todo: implement other metrices
        self.landmarks = landmarks
        self.witnesses = witnesses
        self.metric_computed = False
        self.simplex_tree_computed = False
        self.landmarks_idxs = landmarks_idxs
        self.isomap_eps = isomap_eps

        self.distances = pairwise_distances(witnesses, landmarks, n_jobs = n_jobs)
        if isomap_eps > 0:
            #distances = pairwise_distances(witnesses, n_jobs = -1)
                        
            # todo: optimize
            def _create_large_matrix():
                matrix = np.zeros((self.distances.shape[0], self.distances.shape[0]))
                for i in range(self.distances.shape[0]):
                    for j in range(self.distances.shape[1]):
                        if self.distances[i][j] < self.isomap_eps:
                            matrix[i][landmarks_idxs[j]] = self.distances[i][j]
                return matrix
            def _create_small_matrix(matrix):
                for i in range(self.distances.shape[0]):
                    for j in range(self.distances.shape[1]):
                            self.distances[i][j] = matrix[i][landmarks_idxs[j]]
                
            matrix = _create_large_matrix()
            matrix = csr_matrix(matrix)
            matrix = floyd_warshall(csgraph = matrix, directed = False)
            self.distances_isomap = matrix
            _create_small_matrix(matrix)


    def compute_simplicial_complex(self, d_max, create_metric=False, r_max=None, create_simplex_tree=False, n_jobs = 1):
        if n_jobs == 1:
            self.compute_simplicial_complex_single(d_max=d_max, r_max=r_max)
        else:
            self.compute_simplicial_complex_parallel(d_max=d_max, r_max=r_max, n_jobs=n_jobs)

    def _compute_metric_for_one_witness(self, row):
        sorted_row = sorted([*enumerate(row)], key=lambda x: x[1])
            
        landmark_dist_w = torch.ones(len(self.landmarks), len(self.landmarks))*math.inf
        for element in sorted_row:
            landmark_dist_w[element[0], :] = torch.ones(len(self.landmarks))*element[1]
            landmark_dist_w[:, element[0]] = torch.ones(len(self.landmarks))*element[1]
        return landmark_dist_w

    def compute_metric_optimized(self, n_jobs=1):
        '''
        Computes matrix containing the filtration values for the appearance of 1-simplicies.
        landmark_dist: n_l x n_l
        '''
        assert isinstance(n_jobs, int)

        global _compute_metric_multiprocessing

        def _compute_metric_multiprocessing(distances):
            landmark_dist_process = torch.ones(len(self.landmarks), len(self.landmarks))*math.inf
            for row_i in range(distances.shape[0]):
                row = distances[row_i, :]
                sorted_row = sorted([*enumerate(row)], key=lambda x: x[1])

                landmark_dist_w = torch.ones(len(self.landmarks), len(self.landmarks))*math.inf
                for element in sorted_row:
                    landmark_dist_w[element[0], :] = torch.ones(len(self.landmarks))*element[1]
                    landmark_dist_w[:, element[0]] = torch.ones(len(self.landmarks))*element[1]
                landmark_dist_process = \
                    torch.min(torch.stack((landmark_dist_w, landmark_dist_process)),
                              dim=0)[0]
            return landmark_dist_process

        if n_jobs == 1:
            landmark_dist = torch.ones(len(self.landmarks), len(self.landmarks))*math.inf
            for row_i in range(self.distances.shape[0]):
                row = self.distances[row_i, :]
                landmark_dist_w = self._compute_metric_for_one_witness(row)
                landmark_dist = \
                    torch.min(torch.stack((landmark_dist, landmark_dist_w)),
                              dim=0)[0]
            self.landmarks_dist = landmark_dist
            self.metric_computed = True
        else:
            if n_jobs == -1:
                n_jobs = os.cpu_count()
            mp.set_start_method('fork')
            pool = mp.Pool(processes=n_jobs)
            distances_chunk = np.array_split(self.distances, n_jobs)

            results = pool.map(_compute_metric_multiprocessing, distances_chunk)
            pool.close()
            pool.join()
            
            self.landmarks_dist = torch.min(torch.stack((results)), dim=0)[0]
            self.metric_computed = True

    def compute_1d_simplex_tree(self, r_max = None):
        assert self.metric_computed
        simplex_tree = gudhi.SimplexTree()
        
        for i in range(0, len(self.landmarks)):
            for j in range(i, len(self.landmarks)):
                if r_max is None or float(self.landmarks_dist[i][j]) < r_max:
                    simplex_tree.insert([i, j], float(self.landmarks_dist[i][j]))
                
        self.simplex_tree = simplex_tree
        self.simplex_tree_computed = True
        
    #########################################################################################

    def _update_register_simplex(self, simplicial_complex_temp, i_add, i_dist, max_dim=math.inf):
        simplex_add = []
        for e in simplicial_complex_temp:
            element = e[0]
            if (element[0] != i_add and len(element) == 1) or (1 < len(element) < max_dim+1):
                element_copy = element.copy()
                element_copy.append(i_add)
                simplex_add.append([element_copy, i_dist])
            else:
                pass
        return simplex_add

    def compute_simplicial_complex_single(self, d_max, r_max=None):
        '''
        Computes simplex tree and a matrix containing the filtration values for the appearance of 1-simplicies.
        d_max: max dimension of simplicies in the simplex tree
        r_max: max filtration value
        '''


        simplicial_complex = []
        try:
            simplex_tree = gudhi.SimplexTree()
        except:
            print('Cannot create simplex tree')

        for row_i in range(self.distances.shape[0]):
            row = self.distances[row_i, :]

            # sort row by landmarks witnessed
            sorted_row = sorted([*enumerate(row)], key=lambda x: x[1])
            if r_max != None:
                sorted_row_new_temp = []
                for element in sorted_row:
                    if element[1] < r_max:
                        sorted_row_new_temp.append(element)
                sorted_row = sorted_row_new_temp

            simplices_temp = []
            for i in range(len(sorted_row)):
                simplices_temp.append([[sorted_row[i][0]], sorted_row[i][1]])
                simplex_add = self._update_register_simplex(simplices_temp.copy(), sorted_row[i][0],
                                                            sorted_row[i][1], d_max)
                
                simplices_temp += simplex_add

            simplicial_complex += simplices_temp

            #self.simplicial_complex = simplicial_complex
        sorted_simplicial_compex = sorted(simplicial_complex, key=lambda x: x[1])

        for simplex in sorted_simplicial_compex:
            simplex_tree.insert(simplex[0], filtration=simplex[1])
            self.simplex_tree = simplex_tree
        
        self.simplex_tree_computed = True

    def compute_simplicial_complex_parallel(self, d_max=math.inf, r_max=math.inf, n_jobs=-1):
        
        global process_wc

        def process_wc(distances, r_max=r_max, d_max=d_max):

            simplicial_complex = []

            def update_register_simplex(simplicial_complex, i_add, i_dist, max_dim):
                simplex_add = []
                for e in simplicial_complex:
                    element = e[0]
                    if (element[0] != i_add and len(element) == 1) or (
                            1 < len(element) < max_dim+1):
                        element_copy = element.copy()
                        element_copy.append(i_add)
                        simplex_add.append([element_copy, i_dist])
                    else:
                        pass
                return simplex_add


            for row_i in range(distances.shape[0]):
                row = distances[row_i, :]
                sorted_row = sorted([*enumerate(row)], key=lambda x: x[1])
                if r_max != None:
                    sorted_row_new_temp = []
                    for element in sorted_row:
                        if element[1] < r_max:
                            sorted_row_new_temp.append(element)
                    sorted_row = sorted_row_new_temp

                simplices_temp = []
                for i in range(len(sorted_row)):
                    simplices_temp.append([[sorted_row[i][0]], sorted_row[i][1]])
                    simplex_add = update_register_simplex(simplices_temp.copy(),
                                                          sorted_row[i][0],
                                                          sorted_row[i][1], d_max)
                    
                    simplices_temp += simplex_add

                simplicial_complex += simplices_temp

            return simplicial_complex

        def combine_results(results):

            simplicial_complex = []

            for result in results:
                simplicial_complex += result

            return simplicial_complex

        try:
            simplex_tree = gudhi.SimplexTree()
        except:
            print('Cannot create simplex tree')

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        if True:
            mp.set_start_method('fork')
            pool = mp.Pool(processes=n_jobs)
            distances_chunk = np.array_split(self.distances, n_jobs)

            results = pool.map(process_wc, distances_chunk)

            pool.close()
            pool.join()
            
            simplicial_complex = combine_results(results)
        
            #self.simplicial_complex = simplicial_complex
        sorted_simplicial_compex = sorted(simplicial_complex, key=lambda x: x[1])

        for simplex in sorted_simplicial_compex:
            simplex_tree.insert(simplex[0], filtration=simplex[1])
            self.simplex_tree = simplex_tree
            
        self.simplex_tree_computed = True


    def get_diagram(self, show=False, path_to_save=None):
        assert self.simplex_tree_computed
        fig, ax = plt.subplots()

        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_diagram(diag, axes=ax, legend=True)


        if path_to_save is not None:
            plt.savefig(path_to_save + '/diagram.png', dpi = 200)
        if show:
            plt.show()
        plt.close()
        
    def get_barcode(self, show=False, path_to_save=None):
        assert self.simplex_tree_computed
        fig, ax = plt.subplots()

        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_barcode(diag, axes=ax, legend=True)


        if path_to_save is not None:
            plt.savefig(path_to_save + '/barcode.png', dpi = 200)
        if show:
            plt.show()
        plt.close()
        
    def get_persistence(self, dim=0, from_value=0, to_value=50):
        assert self.simplex_tree_computed
        return self.simplex_tree.persistent_betti_numbers(from_value, to_value)

    def check_distance_matrix(self):
        assert self.metric_computed
        return not np.any(self.landmarks_dist == MAX_DIST_INIT)
        
    def animate_simplex_tree(self, path_to_save):
        assert self.simplex_tree_computed
        gen = self.simplex_tree.get_filtration()
        
        verts = []
        l = list(gen)
        scale = NUMBER_OF_FRAMES/l[-1][1]

        for num in range(1, NUMBER_OF_FRAMES + 1):
            fig = plt.figure()
            ax = fig.add_subplot(projection = "3d")
            if self.witnesses.shape[0] <= MAX_N_PLOT:
                ax.scatter3D(self.witnesses[:MAX_N_PLOT, 0], self.witnesses[:MAX_N_PLOT, 1], self.witnesses[:MAX_N_PLOT, 2], s = 3, linewidths = 0.1)
            ax.scatter3D(self.landmarks[:MAX_N_PLOT, 0], self.landmarks[:MAX_N_PLOT, 1], self.landmarks[:MAX_N_PLOT, 2], s = 6, linewidths = 3, color = 'C1')
            ax.set_xlabel('X, Mpc')
            ax.set_ylabel('Y, Mpc')
            ax.set_zlabel('Z, Mpc')
            for element in l:
                if(element[1]*scale <= num):
                    if(len(element[0]) == 2):
                        x = [self.landmarks[element[0][0]][0], self.landmarks[element[0][1]][0]]
                        y = [self.landmarks[element[0][0]][1], self.landmarks[element[0][1]][1]]
                        z = [self.landmarks[element[0][0]][2], self.landmarks[element[0][1]][2]]
                        ax.plot(x, y, z)
                    if(len(element[0]) == 3):
                        x = [self.landmarks[element[0][0]][0], self.landmarks[element[0][1]][0], self.landmarks[element[0][2]][0]]
                        y = [self.landmarks[element[0][0]][1], self.landmarks[element[0][1]][1], self.landmarks[element[0][2]][1]]
                        z = [self.landmarks[element[0][0]][2], self.landmarks[element[0][1]][2], self.landmarks[element[0][2]][2]]
                        verts.append(list(zip(x, y, z)))
                        poly = Poly3DCollection(verts)
                        poly.set_color(colors.rgb2hex(np.random.rand(3)))
                        ax.add_collection3d(poly)
                        verts.clear()
              
            ax.set_title(f"Animation of witness filtration: picture #{num} of {NUMBER_OF_FRAMES}")
            if path_to_save is not None:
                plt.savefig(path_to_save + f"/picture#{num}.png", dpi = 200)
            plt.show()
            
            
    def tomato(self):
        '''
        Tomato clustering (experimental)
        '''
        t = Tomato()
        t.fit(self.witnesses)
        return t
        
        
            
    
