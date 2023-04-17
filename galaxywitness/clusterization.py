from collections import defaultdict
# from itertools import groupby
import numpy as np
from scipy.spatial.transform import Rotation

import plotly.express as px
import plotly.graph_objects as go

from gudhi.clustering.tomato import Tomato
from gudhi.representations.preprocessing import ProminentPoints

"""
If one wants to compare two clusterizations they should collect the data (lists of coordinates and weights) needed for
the constructor. As a temporary solution this cold be done by adding several changes to `clustering` function
from __main__.py. Insert the following at the end of the function:

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
df = pd.DataFrame(d)
df['weight'] = 1.0
df['cluster'] = tomato.labels_
return df

Also, we still need group points by cluster label, this code does the work:

cl = []
for c in df['cluster'].unique():
    d = df.loc[cl1['cluster'] == c, 'x':'weight'].to_numpy().T.tolist()
    cl.append(d)
o1 = Clusterization(len(cl), cl)

Now you can
cost = o1.compare_clusterization(o2)
print(cost)
"""


def center_of_mass_diff(c1, c2):
    sum_square = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2
    dist = np.sqrt(sum_square)
    weight_diff = abs(c1[3] - c2[3])  # improve weight_diff metric
    return dist + weight_diff


def distances_matrix(centers1, centers2):
    """
    matrix is supposed to have a shape (n, m), n <= m
    """
    n_cntrs = centers2 if len(centers1) > len(centers2) else centers1
    m_cntrs = centers1 if len(centers1) > len(centers2) else centers2
    a = np.zeros((len(n_cntrs) + 1, len(m_cntrs) + 1))
    for i in range(1, a.shape[0]):
        for j in range(1, a.shape[1]):
            a[i][j] = center_of_mass_diff(n_cntrs[i - 1], m_cntrs[j - 1])
    return a


def Hungarian(a):
    n = a.shape[0] - 1  # истинные размеры матрицы расстояний, a.shape учитывает фиктивные нулевые строку и столбец
    m = a.shape[1] - 1
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    matches = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        matches[0] = i
        j_free = 0
        minv = [np.inf] * (m + 1)
        used = [False] * (m + 1)
        while matches[j_free] != 0:
            used[j_free] = True
            i_free = matches[j_free]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i_free][j] - u[i_free] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j_free
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[matches[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j_free = j1
        while j_free != 0:
            j1 = way[j_free]
            matches[j_free] = matches[j1]
            j_free = j1
    cost = -v[0]
    return cost, matches[1:]


class Clusterization:
    """
    Class for handling clusterization of a point cloud

    :param n_clusters: number of clusters.
    :type n_clusters: int

    :param clusters: collection of clusters, each cluster contains four subarrays:
    list of first coordinates, list of second coordinates, list of third coordinates, list of weights.
    Weights are optional to pass though for convenience __init__() fills the parameter with 1.0 in case it isn't present
    :type clusters: list, Iterable, dict, or DataFrame

    :param centers_of_mass_computed: flag is set to False unless centers of mass were computed for clusters.
    :type centers_of_mass_computed: bool

    :param centers_of_mass: collection of computed centers of mass,
    info contained about each center: [coord_1, coord_2, coord_3, weight]
    :type centers_of_mass: list

    """
    __slots__ = [
        'points',
        'labels',
        'n_clusters',
        'clusters',
        'centers_of_mass_computed',
        'centers_of_mass'
    ]

    def __init__(self, points, n_clusters=0, clusters=None):
        """
        Constuctor
        """
        self.points = points
        self.labels = None
        self.n_clusters = n_clusters
        self.clusters = clusters
        self.centers_of_mass_computed = False
        self.centers_of_mass = []

    def center_of_mass(self):
        """
        Compute centroids of clusters in clustering
        """
        for cluster in self.clusters:
            weight = sum(cluster[3])  #
            av_x = sum(np.multiply(cluster[0], cluster[3])) / weight
            av_y = sum(np.multiply(cluster[1], cluster[3])) / weight
            av_z = sum(np.multiply(cluster[2], cluster[3])) / weight
            self.centers_of_mass.append([av_x, av_y, av_z, weight])
        self.centers_of_mass_computed = True
        return self.centers_of_mass

    def _build_clustering(self):
        if self.labels is None: return
        self.clusters = []
        labels = defaultdict(list)
        for i, l in enumerate(self.labels):
            labels[l].append(i)

        for _, cluster in labels.items():
            # temp = np.array(cluster)
            self.clusters.append([self.points[cluster, 0], self.points[cluster, 1], 
                                  self.points[cluster, 2], [1.]*len(cluster)])

        self.n_clusters = len(self.clusters)
        self.center_of_mass()

    def import_clustering(self, labels):
        """
        Import outer clustering

        :param labels: labels of outer clustering
        :type labels: list or np.ndarray
        """
        self.labels = labels
        self._build_clustering()


    def tomato(self, max_fil_val=7.5):
        """
        Tomato clustering

        :param max_fil_val: maximum value of filtration
        :type max_fil_val: float
        """
        tomato = Tomato(density_type='DTM')
        tomato.fit(self.points)
        self.n_clusters = self._compute_number_of_clusters(tomato.diagram_, max_fil_val)
        #tomato.n_clusters_ = self.n_clusters
        self.n_clusters = tomato.n_clusters_
        self.labels = tomato.labels_
        self._build_clustering()


    def _compute_number_of_clusters(self, diag, max_fil_val=7.5):
        new_diag = []

        if diag.size > 0:
            prom = ProminentPoints(use=True, num_pts=len(diag), threshold=max_fil_val)
            new_diag = prom.transform([diag])

        return len(new_diag)

    def draw_projections(self, num):
        """
        Draw projections of clustering of point cloud on the several random planes

        :param num: number of planes
        :type num: int
        """
        thetas = np.random.rand(num)*np.pi
        axes = np.random.randn(3, num)
        axes /= np.linalg.norm(axes, axis=0)
        for j in range(num):
            axis = axes[:, j]
            rotation = Rotation.from_quat([np.sin(thetas[j])*axis[0], np.sin(thetas[j])*axis[1],\
                np.sin(thetas[j])*axis[2], np.cos(thetas[j])])
            points = rotation.apply(self.points)
            fig = px.scatter(x=points[:, 0], y=points[:, 1], color=self.labels)
            fig.update_traces(marker_size=2)
            fig.show()


    def draw_clustering(self):
        """
        Draw clustering of point cloud
        """
        fig = go.Figure(data=[go.Scatter3d(x=self.points[:, 0],
                                           y=self.points[:, 1],
                                           z=self.points[:, 2],
                                           mode='markers',
                                           marker=dict(size=1, color=self.labels))])

        fig.update_layout(scene=dict(xaxis_title="X, Mpc",
                                     yaxis_title="Y, Mpc",
                                     zaxis_title="Z, Mpc"))
        fig.show()


    def compare_clusterization(self, other):
        """
        Compare two clusterizations (not ready)

        :param other: another clustering
        :type num: galaxywitness.clusterization.Clusterization
        """
        if not self.centers_of_mass_computed:
            self.center_of_mass()
        if not other.centers_of_mass_computed:
            other.center_of_mass()

        print(self.n_clusters, other.n_clusters)
        a = distances_matrix(self.centers_of_mass, other.centers_of_mass)
        cost, _ = Hungarian(a)

        return cost
