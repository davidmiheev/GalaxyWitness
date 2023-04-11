import gudhi

from galaxywitness.base_complex import BaseComplex

# hard-coded
MAX_N_PLOT = 10000
NUMBER_OF_FRAMES = 6


class RipsComplex(BaseComplex):
    """
    Main class for handling data about the point cloud and the simplex tree
    of filtered Rips complex

    :param points: set of landmarks in :math:`\mathbb{R}^d`.
    :type points: np.array size of *n_landmarks x 3*

    """

    def __init__(self, points, max_edge_length, sparse=None):
        """
        Constuctor

        """
        super().__init__(points)
        self.max_edge_length = max_edge_length
        self.sparse = sparse
        self.max_dimension = 1  # default value though can be changed

    def compute_simplicial_complex(self, d_max, r_max, **kwargs):
        """
        Compute custom filtered simplicial complex

        :param r_max: max filtration value
        :type  r_max: float

        """

        tmp = gudhi.RipsComplex(points=self.points, max_edge_length=r_max, sparse=self.sparse)

        self.simplex_tree = tmp.create_simplex_tree(max_dimension=self.max_dimension)
        self.simplex_tree.collapse_edges()
        self.simplex_tree.expansion(d_max)
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


    # def tomato(self, den_type, graph):
    #     """
    #     ToMATo clustering with automatic choice of number of clusters.
    #     Hence, clustering depends on filtered complex construction and
    #     max value of filtration.

    #     """
    #     self.graph_type = graph
    #     t = Tomato(density_type=den_type, graph_type=self.graph_type)
    #     if den_type == 'manual' and self.graph_type != 'manual':
    #         t.fit(self.points, weights=self.density_class.foo(self.points))
    #     elif den_type == 'manual' and self.graph_type == 'manual':
    #         t.fit(self.get_adjacency_list(), weights=self.density_class.foo(self.points))
    #     else:
    #         t.fit(self.points)
    #     t.n_clusters_ = self.betti[0]
    #     return t
