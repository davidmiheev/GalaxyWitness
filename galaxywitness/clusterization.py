import numpy as np

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
    a = np.zeros((len(centers1) + 1, len(centers2) + 1))
    for i in range(1, a.shape[0]):
        for j in range(1, a.shape[1]):
            a[i][j] = center_of_mass_diff(centers1[i - 1], centers2[j - 1])
    return a


def Hungarian(a):
    u = [0] * (a.shape[1])
    v = [0] * (a.shape[0])
    matches = [0] * (a.shape[1])
    way = [0] * (a.shape[1])
    for i in range(1, a.shape[0]):
        matches[0] = i
        j_free = 0
        minv = [np.inf] * (a.shape[1])
        used = [False] * (a.shape[1])
        while matches[j_free] != 0:
            used[j_free] = True
            i_free = matches[j_free]
            delta = np.inf
            j1 = 0
            for j in range(1, a.shape[1]):
                if not used[j]:
                    cur = a[i_free][j] - u[i_free] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j_free
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, a.shape[1]):
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
    :type: centers_of_mass: list()

    """
    __slots__ = [
        'n_clusters',
        'clusters',
        'centers_of_mass_computed',
        'centers_of_mass'
    ]

    def __init__(self, n_clusters, clusters):
        self.n_clusters = n_clusters
        self.clusters = clusters
        self.centers_of_mass_computed = False
        self.centers_of_mass = list()

    def center_of_mass(self):
        for cluster in self.clusters:
            weight = sum(cluster[0])  #
            av_x = sum(np.multiply(cluster[0], cluster[3])) / weight
            av_y = sum(np.multiply(cluster[1], cluster[3])) / weight
            av_z = sum(np.multiply(cluster[2], cluster[3])) / weight
            self.centers_of_mass.append([av_x, av_y, av_z, weight])
        self.centers_of_mass_computed = True
        return self.centers_of_mass

    def compare_clusterization(self, other):
        if self.n_clusters != other.n_clusters:
            raise Exception("Cannot compare clusterizations with different sizes")
        if not self.centers_of_mass_computed:
            self.center_of_mass()
        if not other.centers_of_mass_computed:
            other.center_of_mass()

        a = distances_matrix(self.centers_of_mass, other.centers_of_mass)
        cost, matches = Hungarian(a)

        return cost
