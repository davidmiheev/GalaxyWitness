import math
import os

from joblib import Parallel, delayed
from joblib import dump, load

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

# gudhi is needed to construct a simplex tree and to plot the persistence diagram.
import gudhi
from gudhi.clustering.tomato import Tomato

from sklearn.metrics import pairwise_distances

# hard-coded
#MAX_DIST_INIT = 100000
MAX_N_PLOT = 10000
NUMBER_OF_FRAMES = 6

class WitnessComplex():
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
    
    __slots__ = [
        'landmarks',
        'witnesses',
        'distances',
        'distances_isomap',
        'landmarks_idxs',
        'isomap_eps',
        'simplex_tree',
        'simplex_tree_computed',
        'weights',
        'betti'
    ]

    def __init__(self, landmarks, witnesses, landmarks_idxs, n_jobs = -1, isomap_eps = 0):
        """
        Constuctor
        
        """
        #todo: implement other metrices
        self.landmarks = landmarks
        self.witnesses = witnesses
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
            
    def external_simplex_tree(self, simplex_tree):
        """
        Load external filtered simplicial complex (as simplex tree) to WitnessComplex instance
        
        :param simplex_tree: external simplex tree
        :type simplex_tree: gudhi.SimplexTree
        
        """
        
        self.simplex_tree = simplex_tree
        self.simplex_tree_computed = True


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
            if (element[0] != i_add and len(element) == 1) or (1 < len(element) < 2):
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
                                                            sorted_row[i][1])
                
                simplices_temp += simplex_add

            simplicial_complex += simplices_temp

            #self.simplicial_complex = simplicial_complex
        
        sorted_simplicial_compex = sorted(simplicial_complex, key=lambda x: x[1])

        for simplex in sorted_simplicial_compex:
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
        
        def process_wc(distances, ind, r_max=r_max, d_max=d_max):

            simplicial_complex = []
            

            def update_register_simplex(simplicial_complex, i_add, i_dist):
                simplex_add = []
                for e in simplicial_complex:
                    element = e[0]
                    if (element[0] != i_add and len(element) == 1) or (
                            1 < len(element) < 2):
                        element_copy = element.copy()
                        element_copy.append(i_add)
                        simplex_add.append([element_copy, i_dist])
                    else:
                        pass
                return simplex_add


            for row_i in range(distances[ind].shape[0]):
                row = distances[ind][row_i, :]
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
                                                          sorted_row[i][1])
                    
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
        data = load(data_filename_memmap, mmap_mode='r')
        
        results = Parallel(n_jobs=n_jobs)(delayed(process_wc)(distances=distances_chunk, ind=i) for i in range(n_jobs))
        #pool.map(process_wc, distances_chunk)

        #pool.close()
        #pool.join()
          
        simplicial_complex = combine_results(results)
        
        sorted_simplicial_compex = sorted(simplicial_complex, key=lambda x: x[1])

        for simplex in sorted_simplicial_compex:
            simplex_tree.insert(simplex[0], filtration=simplex[1])
            self.simplex_tree = simplex_tree
            
        self.simplex_tree.expansion(d_max)    
        self.simplex_tree_computed = True
        
    #################################################################################
        
    def get_persistence_betti(self, dim, magnitude):
        """
        Computation of persistence betti numbers
        
        :param dim: max dimension of betti numbers
        :type  dim: int
        :param magnitude: level of significance
        :type  magnitude: float
        :return: list of persistence betti numbers for dimensions 0...dim
        :rtype: np.array
        """
        assert self.simplex_tree_computed
        self.simplex_tree.compute_persistence()
        ans = np.zeros(dim, dtype = int)
        for j in range(dim):
            pers = self.simplex_tree.persistence_intervals_in_dimension(j)
            for e in pers:
                if e[1] - e[0] > magnitude:
                    ans[j] += 1
        self.betti = ans        
        return ans


    def get_diagram(self, show=False, path_to_save=None):
        """
        Draw persistent diagram
        
        :param show: show diagram? (Optional)
        :type  show: bool
        :param path_to_save: place, where we are saving files
        :type  path_to_save: str
        """
        
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
        """
        Draw barcode
        
        :param show: show barcode? (Optional)
        :type  show: bool
        :param path_to_save: place, where we are saving files
        :type  path_to_save: str
        
        """
        
        assert self.simplex_tree_computed
        fig, ax = plt.subplots()

        diag = self.simplex_tree.persistence()
        
        gudhi.plot_persistence_barcode(diag, axes=ax, legend=True)

        if path_to_save is not None:
            plt.savefig(path_to_save + '/barcode.png', dpi = 200)
        if show:
            plt.show()
        
        plt.close()

        
    def animate_simplex_tree(self, path_to_save):
        """
        Draw animation of filtration (powered by matplotlib)
        
        :param path_to_save: place, where we are saving files
        :type  path_to_save: str
         
        """
        assert self.simplex_tree_computed
        
        gen = self.simplex_tree.get_filtration()
        
        verts = []
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
            
            for element in gen:
                if(element[1]*scale <= num):
                    if(len(element[0]) == 2):
                        x = [self.landmarks[element[0][0]][0], 
                        self.landmarks[element[0][1]][0]]
                        
                        y = [self.landmarks[element[0][0]][1], 
                        self.landmarks[element[0][1]][1]]
                        
                        z = [self.landmarks[element[0][0]][2], 
                        self.landmarks[element[0][1]][2]]
                        
                        ax.plot(x, y, z)
                        
                    if(len(element[0]) == 3):
                        x = [self.landmarks[element[0][0]][0], 
                        self.landmarks[element[0][1]][0], 
                        self.landmarks[element[0][2]][0]]
                        
                        y = [self.landmarks[element[0][0]][1], 
                        self.landmarks[element[0][1]][1], 
                        self.landmarks[element[0][2]][1]]
                        
                        z = [self.landmarks[element[0][0]][2], 
                        self.landmarks[element[0][1]][2], 
                        self.landmarks[element[0][2]][2]]
                        
                        verts.append(list(zip(x, y, z)))
                        
                        poly = Poly3DCollection(verts)
                        
                        poly.set_color(colors.rgb2hex(np.random.rand(3)))
                        
                        ax.add_collection3d(poly)
                        
                        verts.clear()
              
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
        
        verts = []
        gen = list(gen)
        scale = NUMBER_OF_FRAMES/gen[-1][1]

        for num in range(1, NUMBER_OF_FRAMES + 1):
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
            
            for element in gen:
                if(element[1]*scale <= num):
                    if(len(element[0]) == 2):
                        x = [self.landmarks[element[0][0]][0], 
                        self.landmarks[element[0][1]][0]]
                        
                        y = [self.landmarks[element[0][0]][1], 
                        self.landmarks[element[0][1]][1]]
                        
                        z = [self.landmarks[element[0][0]][2], 
                        self.landmarks[element[0][1]][2]]
                        
                        data.append(go.Scatter3d(x=x,
                                                y=y,
                                                z=z, 
                                                marker = dict(size=2, color='orange'),
                                                line = dict(color=colors.rgb2hex(np.random.rand(3)), width=3)))
                                                
                    if(len(element[0]) == 3):
                        x = [self.landmarks[element[0][0]][0], 
                        self.landmarks[element[0][1]][0], 
                        self.landmarks[element[0][2]][0]]
                        
                        y = [self.landmarks[element[0][0]][1], 
                        self.landmarks[element[0][1]][1], 
                        self.landmarks[element[0][2]][1]]
                        
                        z = [self.landmarks[element[0][0]][2], 
                        self.landmarks[element[0][1]][2], 
                        self.landmarks[element[0][2]][2]]
                        
                        data.append(go.Mesh3d(x=x, 
                                              y=y, 
                                              z=z, 
                                              color=colors.rgb2hex(np.random.rand(3)), 
                                              opacity=0.8))
              
            fig = go.Figure(data=data)
            
            fig.update_layout(scene = dict(xaxis_title = "X, Mpc", 
                                          yaxis_title = "Y, Mpc", 
                                          zaxis_title = "Z, Mpc"))
            
            if path_to_save is not None:
                fig.write_image(path_to_save + f"/picture{num}.pdf")
                
            fig.show()
            
            
            
    def tomato(self):
        """
        ToMATo clustering with automatic choice of number of clusters. 
        Hence clustering depends on filtered complex construction and 
        max value of filtration.
        
        """
        t = Tomato()
        t.fit(self.witnesses)
        t.n_clusters_ = self.betti[0]
        return t
        
        
            
    
