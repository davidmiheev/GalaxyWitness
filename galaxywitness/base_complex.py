from abc import abstractmethod
from collections import defaultdict

import numpy as np
import gudhi
from gudhi.clustering.tomato import Tomato

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors
import plotly.graph_objects as go

from galaxywitness.manual_density import ManualDensity

MAX_N_PLOT = 10000
NUMBER_OF_FRAMES = 6


class BaseComplex:
    """
    Base class for any type of complexes

    """

    # __slots__ = [
    #     'points',
    #     'witnesses',
    #     'distances',
    #     'distances_isomap',
    #     'points_idxs',
    #     'isomap_eps',
    #     'simplex_tree',
    #     'simplex_tree_computed',
    #     'weights',
    #     'betti'
    # ]

    def __init__(self, points=None):
        """
        Constuctor

        """
        self.simplex_tree = None
        self.points = points
        self.betti = None
        self.simplex_tree_computed = False
        self.density_class = ManualDensity()
        # self.graph_type = 'knn'

    @abstractmethod
    def compute_simplicial_complex(self, *args):
        pass

    def external_simplex_tree(self, simplex_tree):
        """
        Load external filtered simplicial complex (as simplex tree) to DescendantComplex instance

        :param simplex_tree: external simplex tree
        :type simplex_tree: gudhi.SimplexTree

        """
        # TODO diversify restructured text for descendant classes

        self.simplex_tree = simplex_tree
        self.simplex_tree_computed = True

    def get_persistence_betti(self, dim, magnitudes):
        """
        Computation of persistence betti numbers

        :param dim: max dimension of betti numbers
        :type  dim: int
        :param magnitudes: levels of significance
        :type  magnitude: list[float]
        :return: list of persistence betti numbers for dimensions 0...dim
        :rtype: np.array
        """
        assert self.simplex_tree_computed
        self.simplex_tree.compute_persistence()
        betti = np.zeros(dim, dtype=int)
        for j in range(dim):
            pers = self.simplex_tree.persistence_intervals_in_dimension(j)
            for e in pers:
                if e[1] - e[0] >= magnitudes[j]:
                    betti[j] += 1
        self.betti = betti
        return betti

    def get_diagram(self, show=False, path_to_save=None):
        """
        Draw persistent diagram

        :param show: show diagram? (Optional)
        :type  show: bool
        :param path_to_save: place, where we are saving files
        :type  path_to_save: str
        """

        assert self.simplex_tree_computed
        _, ax = plt.subplots()

        diag = self.simplex_tree.persistence()
        gudhi.plot_persistence_diagram(diag, axes=ax, legend=True)

        if path_to_save is not None:
            plt.savefig(path_to_save + '/diagram.png', dpi=200)
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
        _, ax = plt.subplots()

        diag = self.simplex_tree.persistence()

        gudhi.plot_persistence_barcode(diag, axes=ax, legend=True)

        if path_to_save is not None:
            plt.savefig(path_to_save + '/barcode.png', dpi=200)
        if show:
            plt.show()

        plt.close()

    def draw_simplicial_complex(self, num, filtration_val, backend, path_to_save=None):
        """
        Draw simplicial complex with filtration value filtration_val

        :param num: number of step
        :type  num: int
        :param filtration_val: filtration value
        :type  filtration_val: float
        :param backend: backend for drawing
        :type  backend: str
        :param path_to_save: place, where we are saving files
        :type  path_to_save: str
        """
        assert self.simplex_tree_computed

        data = []
        if backend == 'mpl':
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            ax.scatter3D(self.points[:MAX_N_PLOT, 0],
                    self.points[:MAX_N_PLOT, 1],
                    self.points[:MAX_N_PLOT, 2],
                    s=1,
                    linewidths=1,
                    color='C1')

            ax.set_xlabel('X, Mpc')
            ax.set_ylabel('Y, Mpc')
            ax.set_zlabel('Z, Mpc')
        elif backend == 'plotly':
            data.append(go.Scatter3d(x=self.points[:MAX_N_PLOT, 0],
                                    y=self.points[:MAX_N_PLOT, 1],
                                    z=self.points[:MAX_N_PLOT, 2],
                                    mode='markers',
                                    marker = dict(size=1, color='blue')))

        gen = self.simplex_tree.get_filtration()

        for edge in gen:
            if edge[1] < filtration_val:
                if len(edge[0]) == 2:
                    x = [self.points[edge[0][0]][0],
                        self.points[edge[0][1]][0]]

                    y = [self.points[edge[0][0]][1],
                        self.points[edge[0][1]][1]]

                    z = [self.points[edge[0][0]][2],
                        self.points[edge[0][1]][2]]

                    if backend == 'mpl':
                        ax.plot(x, y, z, color=colors.rgb2hex(np.random.rand(3)))
                    elif backend == 'plotly':
                        data.append(go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            marker=dict(size=1, color='blue'),
                            line=dict(color=colors.rgb2hex(np.random.rand(3)), width=3)))

                if len(edge[0]) == 3:
                    x = [self.points[edge[0][0]][0],
                        self.points[edge[0][1]][0],
                        self.points[edge[0][2]][0]]

                    y = [self.points[edge[0][0]][1],
                        self.points[edge[0][1]][1],
                        self.points[edge[0][2]][1]]

                    z = [self.points[edge[0][0]][2],
                        self.points[edge[0][1]][2],
                        self.points[edge[0][2]][2]]

                    verts = [list(zip(x, y, z))]

                    if backend == 'mpl':
                        poly = Poly3DCollection(verts)

                        poly.set_color(colors.rgb2hex(np.random.rand(3)))

                        ax.add_collection3d(poly)
                    elif backend == 'plotly':
                        data.append(go.Mesh3d(x=x,
                                             y=y,
                                             z=z,
                                             color=colors.rgb2hex(np.random.rand(3)), opacity=0.5))

        if backend == 'mpl':
            ax.set_title(f"Animation of alpha filtration: picture #{num} of {NUMBER_OF_FRAMES}")

            if path_to_save is not None:
                plt.savefig(path_to_save + f"/picture{num}.png", dpi=200)

            plt.show()
        elif backend == 'plotly':
            fig = go.Figure(data=data)
            fig.update_layout(
                        title=f"Animation of alpha filtration: picture #{num} of {NUMBER_OF_FRAMES}",
                        scene=dict(
                        xaxis_title='X, Mpc',
                        yaxis_title='Y, Mpc',
                        zaxis_title='Z, Mpc'))

            if path_to_save is not None:
                fig.write_image(path_to_save + f"/picture{num}.pdf")

            fig.show()



    def get_adjacency_list(self, max_fil_val):
        """
        Get adjacency list for vertices in 1-skeleton of filtrated simplicial complex
        """
        assert self.simplex_tree_computed
        graph = defaultdict(list)
        for edge, val in self.simplex_tree.get_skeleton(1):
            if val > max_fil_val or len(edge) != 2:
                continue
            graph[edge[0]] += [edge[1]]
            graph[edge[1]] += [edge[0]]

        adj_list = []
        for vertex in range(self.points.shape[0]):
            adj_list.append(graph[vertex])

        return adj_list

    @abstractmethod
    def animate_simplex_tree(self, path_to_save):
        """
        Draw animation of filtrated simplicial complex (powered by matplotlib)

        :param path_to_save: place, where we are saving files
        :type  path_to_save: str

        """


    @abstractmethod
    def animate_simplex_tree_plotly(self, path_to_save):
        """
        Draw animation of filtrated simplicial complex (powered by plotly)

        :param path_to_save: place, where we are saving files
        :type  path_to_save: str
        """


    def tomato(self, max_fil_val=7.5):
        """
        ToMATo clustering with automatic choice of number of clusters.
        Hence, clustering depends on filtered complex construction and
        max value of filtration.

        """
        assert self.simplex_tree_computed
        tomato_clustering = Tomato(graph_type='manual', density_type='manual')
        tomato_clustering.fit(self.get_adjacency_list(max_fil_val), 
                              weights=self.density_class.dtm_density(self.points))
        tomato_clustering.n_clusters_ = self.betti[0]
        return tomato_clustering
