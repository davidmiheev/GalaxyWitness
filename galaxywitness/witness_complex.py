import math
import os

import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
import torch

# dirty fix since gudhi cannot be installed on ETH Zurich cluster
# gudhi is needed to construct a simplex tree and to plot the persistence diagram.
try:
    import gudhi
except:
    print('Failed to import gudhi')
from sklearn.metrics import pairwise_distances

# hard-coded
MAX_DIST_INIT = 1000000


class WitnessComplex():
    __slots__ = [
        'landmarks',
        'witnesses',
        'distances',
        'simplicial_complex',
        'landmarks_dist',
        'simplex_tree',
        'simplex_tree_computed',
        'metric_computed'
    ]

    def __init__(self, landmarks, witnesses, n_jobs = 1):
        #todo: implement other metrices
        self.landmarks = landmarks
        self.witnesses = witnesses
        self.metric_computed = False
        self.simplex_tree_computed = False

        self.distances = pairwise_distances(witnesses, landmarks, n_jobs = n_jobs)


    def compute_simplicial_complex(self, d_max, create_metric=False, r_max=None,create_simplex_tree=False, n_jobs = 1):
        if n_jobs == 1:
            self.compute_simplicial_complex_single(d_max=d_max, create_metric=create_metric, r_max=r_max,create_simplex_tree=create_simplex_tree)
        else:
            self.compute_simplicial_complex_parallel(d_max=d_max, r_max=r_max, create_simplex_tree=create_simplex_tree, create_metric=create_metric,n_jobs=n_jobs)

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
            pool = mp.Pool(processes=n_jobs)
            distances_chunk = np.array_split(self.distances, n_jobs)

            results = pool.map(_compute_metric_multiprocessing, distances_chunk)

            self.landmarks_dist = torch.min(torch.stack((results)), dim=0)[0]
            self.metric_computed = True

    def compute_1d_simplex_tree(self):
        assert self.metric_computed
        simplex_tree = gudhi.SimplexTree()
        
        for i in range(0, len(self.landmarks)):
            for j in range(i, len(self.landmarks)):
                simplex_tree.insert([i, j], float(self.landmarks_dist[i][j]))
                
        self.simplex_tree = simplex_tree
        self.simplex_tree_computed = True
        
    #########################################################################################

    def _update_register_simplex(self, simplicial_complex_temp, i_add, i_dist, max_dim=math.inf):
        simplex_add = []
        for e in simplicial_complex_temp:
            element = e[0]
            if (element[0] is not i_add and len(element) is 1) or (1 < len(element) < max_dim+1):
                element_copy = element.copy()
                element_copy.append(i_add)
                simplex_add.append([element_copy, i_dist])
            else:
                pass
        return simplex_add

    def _update_landmark_dist(self, landmarks_dist, simplex_add):

        for simplex in simplex_add:
            if len(simplex[0]) == 2:
                if landmarks_dist[simplex[0][0]][simplex[0][1]] > simplex[1]:
                    landmarks_dist[simplex[0][0]][simplex[0][1]] = simplex[1]
                    landmarks_dist[simplex[0][1]][simplex[0][0]] = simplex[1]
        return landmarks_dist

    def compute_simplicial_complex_single(self, d_max, create_metric=False, r_max=None,create_simplex_tree=False):
        '''
        Computes simplex tree and a matrix containing the filtration values for the appearance of 1-simplicies.
        d_max: max dimension of simplicies in the simplex tree
        r_max: max filtration value
        '''

        if create_simplex_tree:
            simplicial_complex = []
            try:
                simplex_tree = gudhi.SimplexTree()
            except:
                print('Cannot create simplex tree')

        if create_metric:
            landmarks_dist = np.ones((len(self.landmarks), len(self.landmarks)))*MAX_DIST_INIT

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
                if create_metric:
                    landmarks_dist = self._update_landmark_dist(landmarks_dist, simplex_add)
                simplices_temp += simplex_add

            if create_simplex_tree:
                simplicial_complex += simplices_temp

        if create_metric:
            np.fill_diagonal(landmarks_dist, 0)
            self.landmarks_dist = landmarks_dist
            self.metric_computed = True

        if create_simplex_tree:
            self.simplicial_complex = simplicial_complex
            sorted_simplicial_compex = sorted(simplicial_complex, key=lambda x: x[1])

            for simplex in sorted_simplicial_compex:
                simplex_tree.insert(simplex[0], filtration=simplex[1])
                self.simplex_tree = simplex_tree
            self.simplex_tree_computed = True

    def compute_simplicial_complex_parallel(self, d_max=math.inf, r_max=math.inf,
                                            create_simplex_tree=False, create_metric=False,
                                            n_jobs=-1):
        global process_wc

        def process_wc(distances, r_max=r_max, d_max=d_max, create_metric=create_metric,
                       create_simplex_tree=create_simplex_tree):

            landmarks_dist = np.ones((distances.shape[1], distances.shape[1]))*MAX_DIST_INIT
            simplicial_complex = []

            def update_register_simplex(simplicial_complex, i_add, i_dist, max_dim):
                simplex_add = []
                for e in simplicial_complex:
                    element = e[0]
                    if (element[0] is not i_add and len(element) is 1) or (
                            1 < len(element) < max_dim+1):
                        element_copy = element.copy()
                        element_copy.append(i_add)
                        simplex_add.append([element_copy, i_dist])
                    else:
                        pass
                return simplex_add

            def update_landmark_dist(landmarks_dist, simplex_add):
                for simplex in simplex_add:
                    if len(simplex[0]) == 2:
                        if landmarks_dist[simplex[0][0]][simplex[0][1]] > simplex[1]:
                            landmarks_dist[simplex[0][0]][simplex[0][1]] = simplex[1]
                            landmarks_dist[simplex[0][1]][simplex[0][0]] = simplex[1]
                return landmarks_dist

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
                    if create_metric:
                        landmarks_dist = update_landmark_dist(landmarks_dist, simplex_add)
                    simplices_temp += simplex_add

                if create_simplex_tree:
                    simplicial_complex += simplices_temp

            return landmarks_dist, simplicial_complex

        def combine_results(results, create_metric, create_simplex_tree):

            simplicial_complex = []

            for i, result in enumerate(results):
                if create_metric:
                    if i is 0:
                        landmarks_dist = result[0]
                    else:
                        landmarks_dist = np.dstack((landmarks_dist, result[0]))
                else:
                    landmarks_dist = result[0]

                if create_simplex_tree:
                    simplicial_complex += result[1]

            if create_metric:
                landmarks_dist = np.amin(landmarks_dist, axis=2)
            return simplicial_complex, landmarks_dist

        if create_simplex_tree:
            try:
                simplex_tree = gudhi.SimplexTree()
            except:
                print('Cannot create simplex tree')

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        if True:
            pool = mp.Pool(processes=n_jobs)
            distances_chunk = np.array_split(self.distances, n_jobs)

            results = pool.map(process_wc, distances_chunk)

            pool.close()

            simplicial_complex, landmarks_dist = combine_results(results, create_metric,
                                                                 create_simplex_tree)
        if create_simplex_tree:
            self.simplicial_complex = simplicial_complex
            sorted_simplicial_compex = sorted(simplicial_complex, key=lambda x: x[1])

            for simplex in sorted_simplicial_compex:
                simplex_tree.insert(simplex[0], filtration=simplex[1])
                self.simplex_tree = simplex_tree
            self.simplex_tree_computed = True


        self.simplicial_complex = simplicial_complex
        np.fill_diagonal(landmarks_dist, 0)
        self.landmarks_dist = landmarks_dist
        self.metric_computed = True

    def get_diagram(self, show=False, path_to_save=None):
        assert self.simplex_tree_computed
        fig, ax = plt.subplots()

        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_diagram(diag, axes=ax, legend=True)


        if path_to_save is not None:
            plt.savefig(path_to_save, dpi=200)
        if show:
            plt.show()
        plt.close()
        
    def get_barcode(self, show=False, path_to_save=None):
        assert self.simplex_tree_computed
        fig, ax = plt.subplots()

        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_barcode(diag, axes=ax, legend=True)


        if path_to_save is not None:
            plt.savefig(path_to_save, dpi=200)
        if show:
            plt.show()
        plt.close()

    def check_distance_matrix(self):
        assert self.metric_computed
        return not np.any(self.landmarks_dist == MAX_DIST_INIT)
