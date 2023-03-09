import gudhi
from gudhi.clustering.tomato import Tomato

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

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
        super().__init__()
        self.simplex_tree = None

        self.points = points
        self.simplex_tree_computed = False

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
        verts = []
        gen = list(gen)
        scale = NUMBER_OF_FRAMES / gen[-1][1]

        for num in range(1, NUMBER_OF_FRAMES + 1):
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            ax.scatter3D(self.points[:MAX_N_PLOT, 0],
                         self.points[:MAX_N_PLOT, 1],
                         self.points[:MAX_N_PLOT, 2],
                         s=2,
                         linewidths=1,
                         color='C1')

            ax.set_xlabel('X, Mpc')
            ax.set_ylabel('Y, Mpc')
            ax.set_zlabel('Z, Mpc')

            for element in gen:
                if element[1] * scale <= num:
                    if len(element[0]) == 2:
                        x = [self.points[element[0][0]][0],
                             self.points[element[0][1]][0]]

                        y = [self.points[element[0][0]][1],
                             self.points[element[0][1]][1]]

                        z = [self.points[element[0][0]][2],
                             self.points[element[0][1]][2]]

                        ax.plot(x, y, z)

                    if len(element[0]) == 3:
                        x = [self.points[element[0][0]][0],
                             self.points[element[0][1]][0],
                             self.points[element[0][2]][0]]

                        y = [self.points[element[0][0]][1],
                             self.points[element[0][1]][1],
                             self.points[element[0][2]][1]]

                        z = [self.points[element[0][0]][2],
                             self.points[element[0][1]][2],
                             self.points[element[0][2]][2]]

                        verts.append(list(zip(x, y, z)))

                        poly = Poly3DCollection(verts)

                        poly.set_color(colors.rgb2hex(np.random.rand(3)))

                        ax.add_collection3d(poly)

                        verts.clear()

            ax.set_title(f"Animation of alpha filtration: picture #{num} of {NUMBER_OF_FRAMES}")

            if path_to_save is not None:
                plt.savefig(path_to_save + f"/picture{num}.png", dpi=200)

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
        scale = NUMBER_OF_FRAMES / gen[-1][1]

        for num in range(1, NUMBER_OF_FRAMES + 1):
            data = [go.Scatter3d(x=self.points[:MAX_N_PLOT, 0],
                                 y=self.points[:MAX_N_PLOT, 1],
                                 z=self.points[:MAX_N_PLOT, 2],
                                 mode='markers',
                                 marker=dict(size=2, color='orange'))]

            for element in gen:
                if element[1] * scale <= num:
                    if len(element[0]) == 2:
                        x = [self.points[element[0][0]][0],
                             self.points[element[0][1]][0]]

                        y = [self.points[element[0][0]][1],
                             self.points[element[0][1]][1]]

                        z = [self.points[element[0][0]][2],
                             self.points[element[0][1]][2]]

                        data.append(go.Scatter3d(x=x,
                                                 y=y,
                                                 z=z,
                                                 marker=dict(size=2, color='orange'),
                                                 line=dict(color=colors.rgb2hex(np.random.rand(3)), width=3)))

                    if len(element[0]) == 3:
                        x = [self.points[element[0][0]][0],
                             self.points[element[0][1]][0],
                             self.points[element[0][2]][0]]

                        y = [self.points[element[0][0]][1],
                             self.points[element[0][1]][1],
                             self.points[element[0][2]][1]]

                        z = [self.points[element[0][0]][2],
                             self.points[element[0][1]][2],
                             self.points[element[0][2]][2]]

                        data.append(go.Mesh3d(x=x,
                                              y=y,
                                              z=z,
                                              color=colors.rgb2hex(np.random.rand(3)),
                                              opacity=0.8))

            fig = go.Figure(data=data)

            fig.update_layout(scene=dict(xaxis_title="X, Mpc",
                                         yaxis_title="Y, Mpc",
                                         zaxis_title="Z, Mpc"))

            if path_to_save is not None:
                fig.write_image(path_to_save + f"/picture{num}.pdf")

            fig.show()

    def tomato(self, den_type):
        """
        ToMATo clustering with automatic choice of number of clusters.
        Hence, clustering depends on filtered complex construction and
        max value of filtration.

        """
        t = Tomato(density_type = den_type)
        t.fit(self.points)
        t.n_clusters_ = self.betti[0]
        return t
