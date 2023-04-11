import gudhi

from galaxywitness.base_complex import BaseComplex

# hard-coded
MAX_N_PLOT = 10000
NUMBER_OF_FRAMES = 6


class AlphaComplex(BaseComplex):
    """
    Main class for handling data about the point cloud and the simplex tree
    of filtered alpha complex

    :param points: set of landmarks in :math:`\mathbb{R}^d`.
    :type points: np.array size of *n_landmarks x 3*

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

    def __init__(self, points):
        """
        Constuctor

        """
        super().__init__(points)

    def compute_simplicial_complex(self, r_max, **kwargs):
        """
        Compute custom filtered simplicial complex

        :param r_max: max filtration value
        :type  r_max: float

        """

        tmp = gudhi.AlphaComplex(points=self.points)

        self.simplex_tree = tmp.create_simplex_tree(max_alpha_square=r_max ** 2)
        self.simplex_tree_computed = True

    def animate_simplex_tree(self, path_to_save):
        """
        Draw animation of filtration (powered by matplotlib)

        :param path_to_save: place, where we are saving files
        :type  path_to_save: str

        """
        assert self.simplex_tree_computed

        gen = self.simplex_tree.get_filtration()
        gen = list(gen)
        scale = NUMBER_OF_FRAMES / gen[-1][1]

        for num in range(1, NUMBER_OF_FRAMES + 1):
            self.draw_simplicial_complex(num, num/scale, 'mpl', path_to_save)


    def animate_simplex_tree_plotly(self, path_to_save):
        """
        Draw animation of filtration (powered by plotly)

        :param path_to_save: place, where we are saving files
        :type  path_to_save: str
        """
        assert self.simplex_tree_computed

        gen = self.simplex_tree.get_filtration()
        gen = list(gen)
        scale = NUMBER_OF_FRAMES / gen[-1][1]

        for num in range(1, NUMBER_OF_FRAMES + 1):
            self.draw_simplicial_complex(num, num/scale, 'plotly', path_to_save)
