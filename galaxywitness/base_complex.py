from abc import abstractmethod

import gudhi

import numpy as np

import matplotlib.pyplot as plt


class BaseComplex:
    """
    Base class for any type of Complexes

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

    def __init__(self):
        """
        Constuctor

        """
        self.simplex_tree = None
        self.betti = None
        self.simplex_tree_computed = False

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
