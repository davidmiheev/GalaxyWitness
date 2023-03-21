import math
import os

import multiprocessing as mp
from joblib import Parallel, delayed
from joblib import dump

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

# gudhi is needed to construct a simplex tree and to plot the persistence diagram.
import gudhi
from gudhi.clustering.tomato import Tomato

from sklearn.metrics import pairwise_distances

from galaxywitness.base_complex import BaseComplex

# hard-coded
#MAX_DIST_INIT = 100000
MAX_N_PLOT = 10000
NUMBER_OF_FRAMES = 6


class WitnessComplex(BaseComplex):
    """
    Main class for handling data about the point cloud and the simlex tree
    of filtered witness complex
    
    :param landmarks: set of landmarks in :math:`\mathbb{R}^d`.
    :type landmarks: np.array size of *n_landmarks x 3*
    :param witnesses: set of witnesses in :math:`\mathbb{R}^d`.
    :type witnesses: np.array size of *n_witnesses x 3*
    :param landmarks_idxs: indices of landmarks in witnesses array
    :type landmarks_idxs: np.array[int]

    """
    
    # __slots__ = [
    #     'landmarks',
    #     'witnesses',
    #     'distances',
    #     'distances_isomap',
    #     'landmarks_idxs',
    #     'isomap_eps',
    #     'simplex_tree',
    #     'simplex_tree_computed',
    #     'weights',
    #     'betti'
    # ]

    def __init__(self, landmarks, witnesses, landmarks_idxs, n_jobs = -1, isomap_eps = 0):
        """
        Constuctor
        
        """
        super().__init__(landmarks)
        
        self.landmarks = landmarks
        self.witnesses = witnesses
        self.landmarks_idxs = landmarks_idxs

        self.distances = pairwise_distances(witnesses, landmarks, n_jobs = n_jobs)
            
        if isomap_eps > 0:
            #distances = pairwise_distances(witnesses, n_jobs = -1)
                        
            # todo: optimize
            def create_large_matrix():
                matrix = np.zeros((self.distances.shape[0], self.distances.shape[0]))
                for i in range(self.distances.shape[0]):
                    for j in range(self.distances.shape[1]):
                        if self.distances[i][j] < isomap_eps:
                            matrix[i][landmarks_idxs[j]] = self.distances[i][j]
                return matrix
            def create_small_matrix(matrix):
                for i in range(self.distances.shape[0]):
                    for j in range(self.distances.shape[1]):
                        self.distances[i][j] = matrix[i][landmarks_idxs[j]]
                
            matrix = create_large_matrix()
            matrix = csr_matrix(matrix)
            matrix = floyd_warshall(csgraph = matrix, directed = False)
            self.distances_isomap = matrix
            create_small_matrix(matrix)

    def compute_simplicial_complex(self, d_max, r_max=None, n_jobs = 1, custom=False):
        """
        Compute custom filtered simplicial complex
        
        :param d_max: max dimension of simplicies in the simplex tree 
        :type d_max: int
        :param r_max: max filtration value
        :type  r_max: float
        :param n_jobs: number of threads
        :type  n_jobs: int
        """ 
        if custom:
            if n_jobs == 1:
                self.compute_simplicial_complex_single(d_max=d_max, r_max=r_max)
            else:
                self.compute_simplicial_complex_parallel(d_max=d_max, r_max=r_max, n_jobs=n_jobs)
        else:
            tmp = gudhi.EuclideanStrongWitnessComplex(witnesses=self.witnesses, 
                                                      landmarks=self.landmarks)
                                                                  
            self.simplex_tree = tmp.create_simplex_tree(max_alpha_square=r_max**2,
                                                        limit_dimension=d_max)
            self.simplex_tree_computed = True

    
        
    #########################################################################################

    def _update_register_simplex(self, simplicial_complex_temp, i_add, i_dist):
        simplex_add = []
        for e in simplicial_complex_temp:
            element = e[0]
            if element[0] != i_add and len(element) == 1:
                element_copy = element.copy()
                element_copy.append(i_add)
                simplex_add.append([element_copy, i_dist])
            else:
                pass
        return simplex_add

    def compute_simplicial_complex_single(self, d_max, r_max=None):

        simplicial_complex = []
        
        simplex_tree = gudhi.SimplexTree()

        for row_i in range(self.distances.shape[0]):
            row = self.distances[row_i, :]

            # sort row by landmarks witnessed
            sorted_row = sorted([*enumerate(row)], key=lambda x: x[1])
            if r_max is not None:
                sorted_row_new_temp = []
                for element in sorted_row:
                    if element[1] < r_max:
                        sorted_row_new_temp.append(element)
                sorted_row = sorted_row_new_temp

            simplices_temp = []
            for elem in sorted_row:
                simplices_temp.append([[elem[0]], elem[1]])
                simplex_add = self._update_register_simplex(simplices_temp.copy(), elem[0],
                                                            elem[1])
                
                simplices_temp += simplex_add

            simplicial_complex += simplices_temp

            #self.simplicial_complex = simplicial_complex
        
        sorted_simplicial_complex = sorted(simplicial_complex, key=lambda x: x[1])

        for simplex in sorted_simplicial_complex:
            simplex_tree.insert(simplex[0], filtration=simplex[1])
            self.simplex_tree = simplex_tree
        
        #t = time.time()    
        self.simplex_tree.expansion(d_max)
        #t = time.time() - t
        
        self.simplex_tree_computed = True

    def compute_simplicial_complex_parallel(self, d_max=math.inf, r_max=math.inf, n_jobs=-1):
        #global process_wc
        #@delayed
        #@wrap_non_picklable_objects
        
        def process_wc(distances, ind, r_max=r_max):

            simplicial_complex = []
            

            def update_register_simplex(simplicial_complex, i_add, i_dist):
                simplex_add = []
                for e in simplicial_complex:
                    element = e[0]
                    if element[0] != i_add and len(element) == 1:
                        element_copy = element.copy()
                        element_copy.append(i_add)
                        simplex_add.append([element_copy, i_dist])
                    else:
                        pass
                return simplex_add


            for row_i in range(distances[ind].shape[0]):
                row = distances[ind][row_i, :]
                sorted_row = sorted([*enumerate(row)], key=lambda x: x[1])
                if r_max is not None:
                    sorted_row_new_temp = []
                    for element in sorted_row:
                        if element[1] < r_max:
                            sorted_row_new_temp.append(element)
                    sorted_row = sorted_row_new_temp

                simplices_temp = []
                for elem in sorted_row:
                    simplices_temp.append([[elem[0]], elem[1]])
                    simplex_add = update_register_simplex(simplices_temp.copy(),
                                                          elem[0],
                                                          elem[1])
                    
                    simplices_temp += simplex_add

                simplicial_complex += simplices_temp

            return simplicial_complex
            

        def combine_results(results):
            
            simplicial_complex = []

            for result in results:
                simplicial_complex += result

            return simplicial_complex

        
        simplex_tree = gudhi.SimplexTree()
        
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        #mp.set_start_method('fork')
        #pool = mp.Pool(processes=n_jobs)
        distances_chunk = np.array_split(self.distances, n_jobs)
        folder = './joblib_memmap'
        
        data_filename_memmap = os.path.join(folder, 'distances_memmap')
        dump(distances_chunk, data_filename_memmap)
        # data = load(data_filename_memmap, mmap_mode='r')
        
        results = Parallel(n_jobs=n_jobs)(delayed(process_wc)(distances=distances_chunk, ind=i) for i in range(n_jobs))
        #pool.map(process_wc, distances_chunk)

        #pool.close()
        #pool.join()
          
        simplicial_complex = combine_results(results)
        
        sorted_simplicial_complex = sorted(simplicial_complex, key=lambda x: x[1])

        for simplex in sorted_simplicial_complex:
            simplex_tree.insert(simplex[0], filtration=simplex[1])
            self.simplex_tree = simplex_tree
            
        self.simplex_tree.expansion(d_max)    
        self.simplex_tree_computed = True
        
    #################################################################################

    def animate_simplex_tree(self, path_to_save):
        """
        Draw animation of filtration (powered by matplotlib)
        
        :param path_to_save: place, where we are saving files
        :type  path_to_save: str
         
        """
        assert self.simplex_tree_computed
        
        gen = self.simplex_tree.get_filtration()
        
        data = []
        gen = list(gen)
        scale = NUMBER_OF_FRAMES/gen[-1][1]

        for num in range(1, NUMBER_OF_FRAMES + 1):
            fig = plt.figure()
            ax = fig.add_subplot(projection = "3d")
            if self.witnesses.shape[0] <= MAX_N_PLOT:
                ax.scatter3D(self.witnesses[:MAX_N_PLOT, 0], 
                self.witnesses[:MAX_N_PLOT, 1], 
                self.witnesses[:MAX_N_PLOT, 2], 
                s = 1, 
                linewidths = 0.1)
            ax.scatter3D(self.landmarks[:MAX_N_PLOT, 0], 
                        self.landmarks[:MAX_N_PLOT, 1], 
                        self.landmarks[:MAX_N_PLOT, 2], 
                        s = 2, 
                        linewidths = 1, 
                        color = 'C1')
                        
            ax.set_xlabel('X, Mpc')
            ax.set_ylabel('Y, Mpc')
            ax.set_zlabel('Z, Mpc')

            super().draw_simplicial_complex(ax, data, num/scale)
              
            ax.set_title(f"Animation of witness filtration: picture #{num} of {NUMBER_OF_FRAMES}")
            
            if path_to_save is not None:
                plt.savefig(path_to_save + f"/picture{num}.png", dpi = 200)
                
            plt.show()
            
    def animate_simplex_tree_plotly(self, path_to_save):
        """
        Draw animation of filtration (powered by plotly)
        
        :param path_to_save: place, where we are saving files
        :type  path_to_save: str 
        """
        assert self.simplex_tree_computed
        
        gen = self.simplex_tree.get_filtration()
        
        gen = list(gen)
        scale = NUMBER_OF_FRAMES/gen[-1][1]

        for num in range(1, NUMBER_OF_FRAMES + 1):
            fig = plt.figure()
            ax = fig.add_subplot(projection = "3d")
            data = []
            if self.witnesses.shape[0] <= MAX_N_PLOT:
                data.append(go.Scatter3d(x=self.witnesses[:MAX_N_PLOT, 0], 
                                        y=self.witnesses[:MAX_N_PLOT, 1], 
                                        z=self.witnesses[:MAX_N_PLOT, 2], 
                                        mode='markers', 
                                        marker=dict(size=1, color='blue')))
            
            data.append(go.Scatter3d(x=self.landmarks[:MAX_N_PLOT, 0], 
                                    y=self.landmarks[:MAX_N_PLOT, 1], 
                                    z=self.landmarks[:MAX_N_PLOT, 2], 
                                    mode='markers',
                                    marker=dict(size=2, color='orange')))
            
            super().draw_simplicial_complex(ax, data, num/scale)
              
            fig = go.Figure(data=data)
            
            fig.update_layout(scene = dict(xaxis_title = "X, Mpc", 
                                          yaxis_title = "Y, Mpc", 
                                          zaxis_title = "Z, Mpc"))
            
            if path_to_save is not None:
                fig.write_image(path_to_save + f"/picture{num}.pdf")
                
            fig.show()

    def tomato(self, den_type):
        """
        ToMATo clustering with automatic choice of number of clusters. 
        Hence clustering depends on filtered complex construction and 
        max value of filtration.
        
        """
        t = Tomato(density_type = den_type)
        t.fit(self.witnesses)
        t.n_clusters_ = self.betti[0]
        return t
        
        
