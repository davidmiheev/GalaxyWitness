import pytest
from abc import abstractmethod

import gudhi

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors
from galaxywitness.tests import betti_array
from galaxywitness.manual_density import ManualDensity

import plotly.graph_objects as go


class BaseComplex:
    """
    Base class for any type of Complexes

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
        betti = np.zeros(dim, dtype=int)
        for j in range(dim):
            pers = self.simplex_tree.persistence_intervals_in_dimension(j)
            for e in pers:
                if e[1] - e[0] > magnitude:
                    betti[j] += 1
        self.betti = betti
        betti_array(self.betti)
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

    def draw_simplicial_complex(self, ax, data, filtration_val):
        assert self.simplex_tree_computed

        gen = self.simplex_tree.get_filtration()

        for elem in gen:
                if elem[1] < filtration_val:
                    if len(elem[0]) == 2:
                        x = [self.points[elem[0][0]][0], 
                            self.points[elem[0][1]][0]]
                        
                        y = [self.points[elem[0][0]][1], 
                            self.points[elem[0][1]][1]]
                        
                        z = [self.points[elem[0][0]][2], 
                            self.points[elem[0][1]][2]]
                        
                        ax.plot(x, y, z, color=colors.rgb2hex(np.random.rand(3)), linewidth=3)

                        data.append(go.Scatter3d(x=x,
                                                y=y,
                                                z=z, 
                                                marker = dict(size=2, color='orange'),
                                                line = dict(color=colors.rgb2hex(np.random.rand(3)), width=3)))
                                                
                    if len(elem[0]) == 3:
                        x = [self.points[elem[0][0]][0], 
                        self.points[elem[0][1]][0], 
                        self.points[elem[0][2]][0]]
                        
                        y = [self.points[elem[0][0]][1], 
                        self.points[elem[0][1]][1], 
                        self.points[elem[0][2]][1]]
                        
                        z = [self.points[elem[0][0]][2], 
                        self.points[elem[0][1]][2], 
                        self.points[elem[0][2]][2]]

                        verts = [list(zip(x, y, z))]

                        poly = Poly3DCollection(verts)

                        poly.set_color(colors.rgb2hex(np.random.rand(3)))

                        ax.add_collection3d(poly)
                        
                        data.append(go.Mesh3d(x=x, 
                                              y=y, 
                                              z=z, 
                                              color=colors.rgb2hex(np.random.rand(3)), 
                                              opacity=0.5))

    @abstractmethod
    def animate_simplex_tree(self, path_to_save):
        """
        Draw animation of filtration (powered by matplotlib)

        :param path_to_save: place, where we are saving files
        :type  path_to_save: str

        """


    @abstractmethod
    def animate_simplex_tree_plotly(self, path_to_save):
        """
        Draw animation of filtration (powered by plotly)

        :param path_to_save: place, where we are saving files
        :type  path_to_save: str
        """

    @abstractmethod
    def tomato(self, density_type):
        """
        ToMATo clustering with automatic choice of number of clusters.
        Hence, clustering depends on filtered complex construction and
        max value of filtration.

        """
